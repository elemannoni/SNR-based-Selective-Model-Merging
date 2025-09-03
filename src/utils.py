import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

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

def git_rebasin_align(model_ref, model_to_align):
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
    """Verifica se due modelli hanno pesi identici."""
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True
