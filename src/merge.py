import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import math

def ties_merging_layer(base_param, param_a, param_b):
    """
    Esegue il TIES-merging per un singolo layer (tensore di pesi o bias)
    """
    #Calcolo della differenza rispetto al backbone
    delta_a = param_a - base_param
    delta_b = param_b - base_param

    #Elect Sign: crea una maschera per i segni non conflittuali (un conflitto si ha quando i segni di delta_a e delta_b sono opposti)
    sign_agreement = torch.sign(delta_a) == torch.sign(delta_b)

    #Media dei delta
    merged_delta = (delta_a + delta_b) / 2

    #Applicazione della maschera: vengono azzerati i delta dove i segni sono in conflitto
    merged_delta[~sign_agreement] = 0

    #Applicazione del delta fuso al parametro del backbone per ottenere il nuovo parametro
    return base_param + merged_delta


def slerp_merging(p0: torch.Tensor, p1: torch.Tensor, t: float, epsilon=1e-8) -> torch.Tensor:
    """
    Esegue il merge tramite Spherical Linear Interpolation (SLERP)
    """
    #Salva la forma originale e appiattisce i tensori
    original_shape = p0.shape
    p0 = p0.flatten()
    p1 = p1.flatten()

    #Calcolo delle norme e interpolazione lineare
    norm0 = torch.norm(p0)
    norm1 = torch.norm(p1)
    final_norm = (1 - t) * norm0 + t * norm1

    #Normalizzazione dei vettori per ottenere solo la direzione
    p0_normed = p0 / (norm0 + epsilon)
    p1_normed = p1 / (norm1 + epsilon)

    #Calcolo dell'angolo tra i due versori
    dot_product = torch.dot(p0_normed, p1_normed).clamp(-1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    #Se l'angolo è molto piccolo -> interpolazione lineare
    if sin_omega < epsilon:
        p_final = (1 - t) * p0 + t * p1
    else:
        #Altrimenti interpolaione sferica
        c0 = torch.sin((1 - t) * omega) / sin_omega
        c1 = torch.sin(t * omega) / sin_omega
        p_interp_direction = c0 * p0_normed + c1 * p1_normed
        p_final = p_interp_direction * final_norm

    return p_final.reshape(original_shape)


def merge_models_top_p(model_a, model_b, base_model, snr_a, snr_b, snr_base,
                       top_p=0.25, merge_method='lerp', snr_avg=False):
    """
    Merging dei modelli A e B. Per 'ties', la testa di classificazione finale viene esclusa.
    """
    merged_model = deepcopy(base_model)
    num_layers_to_consider = len(merged_model.layers)
    if merge_method == 'ties':
        num_layers_to_consider -= 1
        
    num_hidden_layers = len(model_a.layers)

    k = math.ceil(num_layers_to_consider * top_p)

    if snr_avg:
        avg_snrs = [(snr_a[i] + snr_b[i]) / 2 for i in range(num_hidden_layers)]
    else:
        avg_snrs = snr_base


    indices_to_merge = np.argsort(avg_snrs[:num_layers_to_consider])[-k:]

    with torch.no_grad():
        for i in range(num_layers_to_consider):
            layer_a, layer_b, base_layer, merged_layer = (
                model_a.layers[i], model_b.layers[i], 
                base_model.layers[i], merged_model.layers[i]
            )

            if i in indices_to_merge:
                merge_alpha = snr_b[i] / (snr_b[i] + snr_a[i])

                if merge_method == 'lerp':
                    merged_layer.weight.data = (1 - merge_alpha) * layer_a.weight.data + merge_alpha * layer_b.weight.data
                    if merged_layer.bias is not None:
                        merged_layer.bias.data = (1 - merge_alpha) * layer_a.bias.data + merge_alpha * layer_b.bias.data
                
                elif merge_method == 'slerp':
                    merged_layer.weight.data = slerp_merging(layer_a.weight.data, layer_b.weight.data, merge_alpha)
                    if merged_layer.bias is not None:
                        merged_layer.bias.data = slerp_merging(layer_a.bias.data, layer_b.bias.data, merge_alpha)
                
                elif merge_method == 'ties':
                    merged_layer.weight.data = ties_merging_layer(
                        base_layer.weight.data,
                        layer_a.weight.data,
                        layer_b.weight.data
                    )
                    if merged_layer.bias is not None:
                        merged_layer.bias.data = ties_merging_layer(
                            base_layer.bias.data,
                            layer_a.bias.data,
                            layer_b.bias.data
                        )
                else:
                    raise ValueError(f"Metodo di fusione '{merge_method}' non supportato.")

            else:
                merged_layer.weight.data = base_layer.weight.data
                if merged_layer.bias is not None:
                    merged_layer.bias.data = base_layer.bias.data
    

    return indices_to_merge, merged_model
