import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import pandas as pd

def transfer_body_weights(backbone_model, expert_model):
    """
    Funzione che copia i pesi dal backbone all'esperto per tutti i layer tranne che per l'output_layer
    """

    backbone_state_dict = backbone_model.state_dict()
    weights_to_load = {}
    for name, param in backbone_state_dict.items():
        if "output_layer" not in name:
            weights_to_load[name] = param

    expert_model.load_state_dict(weights_to_load, strict=False)
    print(f"Pesi del corpo trasferiti da backbone a esperto (output_size={expert_model.output_layer.out_features}).")

def git_rebasin_align(model_ref, model_to_align, device):
    """
    Allinea globalmente i pesi di 'model_to_align' a quelli di 'model_ref' sfruttando l'algoritmo di Git-Rebasin
    """
    aligned_model = deepcopy(model_to_align)
    layers_ref = model_ref.layers
    layers_aligned = aligned_model.layers

    current_permutation = None

    with torch.no_grad():
        for i in range(len(layers_ref)):
            layer_ref = layers_ref[i]
            layer_aligned = layers_aligned[i]
            
            if current_permutation is not None:
                if isinstance(layer_aligned, nn.Linear):
                    layer_aligned.weight.data = layer_aligned.weight.data[:, current_permutation]
                elif isinstance(layer_aligned, nn.Conv2d):
                    if layer_aligned.groups > 1:
                         layer_aligned.weight.data = layer_aligned.weight.data[current_permutation, :, :, :]
                    else:
                         layer_aligned.weight.data = layer_aligned.weight.data[:, current_permutation, :, :]


            if isinstance(layer_aligned, (nn.BatchNorm1d, nn.BatchNorm2d)):
                continue

            
            w_ref = layer_ref.weight.data
            w_aligned = layer_aligned.weight.data
            
            
            w_ref_reshaped = w_ref.view(w_ref.size(0), -1)
            w_aligned_reshaped = w_aligned.view(w_aligned.size(0), -1)
            
            cost_matrix = torch.cdist(w_ref_reshaped, w_aligned_reshaped, p=2.0)
            _, aligned_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
            aligned_indices = torch.tensor(aligned_indices, dtype=torch.long, device=w_ref.device)

            if isinstance(layer_aligned, nn.Linear):
                layer_aligned.weight.data = w_aligned[aligned_indices, :]
            else:
                layer_aligned.weight.data = w_aligned[aligned_indices, :, :, :]
            if layer_aligned.bias is not None:
                layer_aligned.bias.data = layer_aligned.bias.data[aligned_indices]

            if i > 0 and isinstance(layers_aligned[i-1], (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_layer = layers_aligned[i-1]
                bn_layer.weight.data = bn_layer.weight.data[current_permutation]
                bn_layer.bias.data = bn_layer.bias.data[current_permutation]
                bn_layer.running_mean.data = bn_layer.running_mean.data[current_permutation]
                bn_layer.running_var.data = bn_layer.running_var.data[current_permutation]

            is_conv_to_linear = (i + 1 < len(layers_ref) and
                                 isinstance(layer_ref, nn.Conv2d) and
                                 isinstance(layers_ref[i+1], nn.Linear))
            
            if is_conv_to_linear:
                next_layer_aligned = layers_aligned[i+1]
                spatial_dims = next_layer_aligned.in_features // w_ref.size(0)
                
                expanded_indices = torch.repeat_interleave(aligned_indices, spatial_dims)
                offset = torch.arange(0, next_layer_aligned.in_features, spatial_dims, device=w_ref.device)
                offset = torch.repeat_interleave(offset, spatial_dims)
                perm_map = torch.zeros_like(expanded_indices)
                for j, idx in enumerate(aligned_indices):
                    perm_map[j*spatial_dims:(j+1)*spatial_dims] = torch.arange(idx*spatial_dims, (idx+1)*spatial_dims, device=w_ref.device)
                current_permutation = perm_map
            else:
                current_permutation = aligned_indices
    return aligned_model

def cycle_consistency(model, model_A, model_B):
    """
    Calcola la consistenza ciclica riallineando i modelli e confrontando con i pesi originali
    """
    aligned_model_B = git_rebasin_align(model, model_B)
    B_realigned = git_rebasin_align(model_B, aligned_model_B)
    aligned_model_A = git_rebasin_align(model, model_A)
    A_realigned = git_rebasin_align(model_A, aligned_model_A)
    
    cycle_loss = 0
    for layer_orig, layer_realigned in zip(model_A.layers, A_realigned.layers):
      cycle_loss += F.mse_loss(layer_orig.weight.data, layer_realigned.weight.data)
    
    print(f"Perdita di Coerenza Ciclica A: {cycle_loss.item()}")
    
    cycle_loss = 0
    for layer_orig, layer_realigned in zip(model_B.layers, B_realigned.layers):
      cycle_loss += F.mse_loss(layer_orig.weight.data, layer_realigned.weight.data)
    
    print(f"Perdita di Coerenza Ciclica B: {cycle_loss.item()}")


def check_models_identical(m1, m2):
    """
    Verifica se due modelli hanno pesi identici
    """
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True

def calculate_delta(model, base_model, device layer_to_ignore='output'):
    """
    Calcola la differenza (delta) tra i pesi di un modello e un modello base
    """
    base_weights = base_model.state_dict()
    model_weights = model.state_dict()
    delta = {
        key: model_weights[key].to(base_weights[key].device) - base_weights[key]
        for key in base_weights
        if layer_to_ignore not in key
    }
    return delta

def calculate_cosine_similarity(delta1, delta2):
    """
    Calcola la similarità cosenica tra due delta di pesi
    """
    keys = sorted(list(delta1.keys()))
    if not keys: return 0.0
    flat_delta1 = torch.cat([delta1[key].flatten() for key in keys])
    flat_delta2 = torch.cat([delta2[key].flatten() for key in keys])
    similarity = F.cosine_similarity(flat_delta1.float(), flat_delta2.float(), dim=0)
    return similarity.item()

def perform_consistency_analysis(delta_A, delta_B, delta_merged):
    """
    Calcola la consistenza ciclica su un set di delta
    """
    if not delta_merged: #Se non ci sono layer da analizzare
        return 0.0, 0.0, 0.0

    #Vettori di spostamento
    recovered_B_direction = {key: delta_merged[key] - delta_A[key] for key in delta_merged}
    recovered_A_direction = {key: delta_merged[key] - delta_B[key] for key in delta_merged}

    #Vettori target
    target_B_direction = {key: delta_B[key] - delta_A[key] for key in delta_merged}
    target_A_direction = {key: delta_A[key] - delta_B[key] for key in delta_merged}

    #Calcolo similarità
    consistency_A = calculate_cosine_similarity(recovered_A_direction, target_A_direction)
    consistency_B = calculate_cosine_similarity(recovered_B_direction, target_B_direction)
    avg_consistency = (consistency_A + consistency_B) / 2
    
    return consistency_A, consistency_B, avg_consistency

def print_summary(title, results_dict):
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)
    if not results_dict:
        print("Nessun risultato da visualizzare.")
        return
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df = results_df.sort_values(by='avg_consistency', ascending=False)
    print(results_df.to_string(formatters={
        'consistency_A': '{:,.4f}'.format,
        'consistency_B': '{:,.4f}'.format,
        'avg_consistency': '{:,.4f}'.format
    }))

def freeze_layers_selectively(model, trainable_indices: list):
    """  
    Congela tutti i layer del modello tranne l'output_layer e quelli i cui
    indici sono specificati nella lista trainable_indices
    """
    #Congela tutti i parametri
    for param in model.parameters():
        param.requires_grad = False

    #Scongela i layer di trainable_indices
    for i, layer in enumerate(model.layers):
        if i in trainable_indices:
            for param in layer.parameters():
                param.requires_grad = True

    #Scongela sempre l'output_layer
    for param in model.output_layer.parameters():
        param.requires_grad = True

def get_group_name(model_name):
    """
    Estrae il nome del gruppo da un nome di modello completo (il gruppo è tutto ciò che precede la prima parentesi '(')
    """
    if '(' in model_name:
        return model_name.split('(')[0].strip()
    else:
        return model_name.strip()


def get_plot_colors(labels):
    color_map = []
    for name in labels:
        if 'Base Model' in name: color_map.append('gray')
        elif 'Naive Average' in name: color_map.append('slateblue')
        elif 'Expert' in name: color_map.append('silver')
        elif 'Selective FT' in name: color_map.append('lightcoral')
        elif 'Only Head' in name: color_map.append('orange')
        else: color_map.append('mediumseagreen') #Full FT
    return color_map

def extract_metrics_from_results(results, baseline_model_name="Base Model + Fine-Tune"):
    """
    Estrae accuracy, F1, tempo e memoria dal dizionario dei risultati e identifica i valori baseline e zero-shot
    Ritorna un dizionario che contiene liste di etichette e valori per ogni metrica
    """
    labels_acc, data_acc = [], []
    labels_f1, data_f1 = [], []
    labels_time, data_time = [], []
    labels_mem, data_mem = [], []

    baseline_acc, baseline_f1, baseline_time, baseline_mem = None, None, None, None
    min_baseline_acc, min_baseline_f1 = None, None

    for name, res in results.items():
        if res.get('acc_finetuned') is not None:
            labels_acc.append(name)
            data_acc.append(res['acc_finetuned'])
            if name == baseline_model_name:
                baseline_acc = res['acc_finetuned']
                min_baseline_acc = res['acc_zero_shot']
            if "(Head Only)" in name:
                labels_acc.append(name[:-11] + " (Zero-Shot)")
                data_acc.append(res['acc_zero_shot'])

        if res.get('f1_finetuned') is not None:
            labels_f1.append(name)
            data_f1.append(res['f1_finetuned'])
            if name == baseline_model_name:
                baseline_f1 = res['f1_finetuned']
                min_baseline_f1 = res['f1_zero_shot']
            if "(Head Only)" in name:
                labels_f1.append(name[:-11] + " (Zero-Shot)")
                data_f1.append(res['f1_zero_shot'])

        if res.get('time') is not None:
            labels_time.append(name)
            data_time.append(res['time'])
            if name == baseline_model_name:
                baseline_time = res['time']

        if res.get('ram_gb') is not None:
            labels_mem.append(name)
            data_mem.append(res['ram_gb'])
            if name == baseline_model_name:
                baseline_mem = res['ram_gb']

    return {
        "acc": (labels_acc, data_acc, baseline_acc, min_baseline_acc),
        "f1": (labels_f1, data_f1, baseline_f1, min_baseline_f1),
        "time": (labels_time, data_time, baseline_time),
        "mem": (labels_mem, data_mem, baseline_mem)
    }
