import torch
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
