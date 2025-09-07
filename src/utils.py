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
        if backbone_state_dict[name].shape == expert_model.state_dict()[name].shape:
            weights_to_load[name] = param
        else:
            print(f"Salto il layer '{name}' a causa di una forma incompatibile: backbone {param.shape} vs expert {expert_model.state_dict()[name].shape}")

    expert_model.load_state_dict(weights_to_load, strict=False)
    print(f"Pesi del corpo trasferiti da backbone a esperto (output_size={expert_model.fc.out_features}).")

def align_layer_weights(layer_ref, layer_aligned, perm_in=None, perm_out=None):
    """Funzione helper per allineare un singolo layer (Conv o Linear)."""
    with torch.no_grad():
        if perm_in is not None:
            if isinstance(layer_aligned, nn.Linear):
                layer_aligned.weight.data = layer_aligned.weight.data[:, perm_in]
            elif isinstance(layer_aligned, nn.Conv2d):
                layer_aligned.weight.data = layer_aligned.weight.data[:, perm_in, :, :]

        if perm_out is False:
            return None


        w_ref = layer_ref.weight.data
        w_aligned = layer_aligned.weight.data
        
        w_ref_reshaped = w_ref.view(w_ref.size(0), -1)
        w_aligned_reshaped = w_aligned.view(w_aligned.size(0), -1)

        cost_matrix = torch.cdist(w_ref_reshaped, w_aligned_reshaped, p=2.0)
        _, perm_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        perm_indices = torch.tensor(perm_indices, dtype=torch.long, device=w_ref.device)

        layer_aligned.weight.data = w_aligned[perm_indices, :] if isinstance(layer_aligned, nn.Linear) else w_aligned[perm_indices, :, :, :]
        if layer_aligned.bias is not None:
            layer_aligned.bias.data = layer_aligned.bias.data[perm_indices]
            
        return perm_indices

def align_bn_layer(bn_layer, perm):
    """Funzione helper per permutare un layer BatchNorm."""
    if bn_layer is not None and perm is not None:
        bn_layer.weight.data = bn_layer.weight.data[perm]
        bn_layer.bias.data = bn_layer.bias.data[perm]
        bn_layer.running_mean.data = bn_layer.running_mean.data[perm]
        bn_layer.running_var.data = bn_layer.running_var.data[perm]

def git_rebasin_align(model_ref, model_to_align, device):
    """
    Allinea i pesi di 'model_to_align' a 'model_ref' per un'architettura ResNet.
    """
    aligned_model = deepcopy(model_to_align)
    
    perm = align_layer_weights(model_ref.conv1, aligned_model.conv1)
    align_bn_layer(aligned_model.bn1, perm)

    for layer_name in ["layer1", "layer2", "layer3"]:
        layer_ref_group = getattr(model_ref, layer_name)
        layer_aligned_group = getattr(aligned_model, layer_name)
        
        for block_idx in range(len(layer_ref_group)):
            block_ref = layer_ref_group[block_idx]
            block_aligned = layer_aligned_group[block_idx]
            
            align_layer_weights(block_ref.conv1, block_aligned.conv1, perm_in=perm, perm_out=False)
            if len(block_aligned.downsample) > 0:
                align_layer_weights(block_ref.downsample[0], block_aligned.downsample[0], perm_in=perm, perm_out=False)

            perm_conv1 = align_layer_weights(block_ref.conv1, block_aligned.conv1)
            align_bn_layer(block_aligned.bn1, perm_conv1)
            
            perm_conv2 = align_layer_weights(block_ref.conv2, block_aligned.conv2, perm_in=perm_conv1)
            align_bn_layer(block_aligned.bn2, perm_conv2)
            
            if len(block_aligned.downsample) > 0:
                perm_downsample = align_layer_weights(block_ref.downsample[0], block_aligned.downsample[0])
                align_bn_layer(block_aligned.downsample[1], perm_downsample)
            
            perm = perm_conv2

    align_layer_weights(model_ref.fc, aligned_model.fc, perm_in=perm)

    return aligned_model

def cycle_consistency(model, model_A, model_B, device):
    """
    Calcola la consistenza ciclica riallineando i modelli e confrontando con i pesi originali
    """
    aligned_model_B = git_rebasin_align(deepcopy(model), model_B, device)
    B_realigned = git_rebasin_align(deepcopy(model_B), aligned_model_B, device)
    aligned_model_A = git_rebasin_align(deepcopy(model), model_A, device)
    A_realigned = git_rebasin_align(deepcopy(model_A), aligned_model_A, device)

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


def calculate_delta(model, base_model, layer_to_ignore=None):
    """
    Calcola la differenza (delta) tra i pesi, ignorando correttamente
    i layer le cui dimensioni non corrispondono.
    """
    base_weights = base_model.state_dict()
    model_weights = model.state_dict()
    delta = {}
    for key in base_weights:
        if layer_to_ignore and key.startswith(layer_to_ignore):
            continue
        if key in model_weights and base_weights[key].shape == model_weights[key].shape:
            delta[key] = model_weights[key].to(base_weights[key].device) - base_weights[key]
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
    Calcola la consistenza ciclica, operando solo sui layer comuni a tutti i delta.
    """
    common_keys = delta_A.keys() & delta_B.keys() & delta_merged.keys()

    if not common_keys:
        return 0.0, 0.0, 0.0


    recovered_B_direction = {key: delta_merged[key] - delta_A[key] for key in common_keys}
    recovered_A_direction = {key: delta_merged[key] - delta_B[key] for key in common_keys}

    target_B_direction = {key: delta_B[key] - delta_A[key] for key in common_keys}
    target_A_direction = {key: delta_A[key] - delta_B[key] for key in common_keys}

    consistency_A = calculate_cosine_similarity(recovered_A_direction, target_A_direction)
    consistency_B = calculate_cosine_similarity(recovered_B_direction, target_B_direction)
    avg_consistency = (consistency_A + consistency_B) / 2
    
    return consistency_A, consistency_B, avg_consistency

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
    for param in model.fc.parameters():
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
        elif 'Zero-Shot' in name: color_map.append('gold')
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
            if name == baseline_model_name:
                baseline_acc = res['acc_finetuned']
                min_baseline_acc = res['acc_zero_shot']
            if "(Only Head)" in name:
                labels_acc.append(name[:-11] + " (Zero-Shot)")
                data_acc.append(res['acc_zero_shot'])
            labels_acc.append(name)
            data_acc.append(res['acc_finetuned'])

        if res.get('f1_finetuned') is not None:
            if name == baseline_model_name:
                baseline_f1 = res['f1_finetuned']
                min_baseline_f1 = res['f1_zero_shot']
            if "(Only Head)" in name:
                labels_f1.append(name[:-11] + " (Zero-Shot)")
                data_f1.append(res['f1_zero_shot'])
            labels_f1.append(name)
            data_f1.append(res['f1_finetuned'])

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