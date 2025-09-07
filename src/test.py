import time
import torch
import tracemalloc
from copy import deepcopy

from .train_and_evaluation import train_model, evaluate_model
from .utils import freeze_layers_selectively, transfer_body_weights

def finetuning_experiments(model, model_A, model_B, merged_models_to_evaluate, loader_full_train, loader_full_test, device, is_alignment_noop, 
                               epochs_base=10, lr_base=1e-4, epochs_merged=10, lr_merged=1e-4):
  results = {}

  print("\nTest: Fine-Tuning del Modello Base (No-Merge)...")
  base_ft_model = deepcopy(model)
  acc_before, f1_before = evaluate_model(base_ft_model, loader_full_test, device)

  start_time = time.time()
  if device == "cuda": torch.cuda.reset_peak_memory_stats(device)
  else: tracemalloc.start()

  train_model(base_ft_model, loader_full_train, loader_full_test, device, lr=lr_base, epochs=epochs_base)

  if device == "cuda": peak_ram_bytes = torch.cuda.max_memory_allocated(device)
  else: peak_ram_bytes = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
  end_time = time.time()

  acc_after, f1_after = evaluate_model(base_ft_model, loader_full_test, device)
  results["Base Model + Fine-Tune"] = {
      'acc_zero_shot': acc_before,
      'acc_finetuned': acc_after,
      'f1_zero_shot': f1_before,
      'f1_finetuned': f1_after,
      'time': end_time - start_time,
      'ram_gb': peak_ram_bytes / 1024**3,
      'details': '100% dei parametri addestrati.'
  }

  print("\nTest: Fusione Naive (Senza Allineamento)...")
  if not is_alignment_noop:
      naive_merged_model = deepcopy(model)
      model_A_state_dict = model_A.state_dict()
      model_B_state_dict = model_B.state_dict()
      naive_merged_state_dict = naive_merged_model.state_dict()
      for key in model_A_state_dict:
          if model_A_state_dict[key].shape == model_B_state_dict[key].shape and model_A_state_dict[key].shape == naive_merged_state_dict[key].shape:
              naive_merged_state_dict[key] = (model_A_state_dict[key] + model_B_state_dict[key]) / 2.0
          else:
              print(f" -> Skipping key '{key}' due to shape mismatch...")
      naive_merged_model.load_state_dict(naive_merged_state_dict)

      acc_before_naive, f1_before_naive = evaluate_model(naive_merged_model, loader_full_test, device)

      start_time = time.time()
      if device == "cuda": torch.cuda.reset_peak_memory_stats(device)
      else: tracemalloc.start()

      train_model(naive_merged_model, loader_full_train, loader_full_test, device, lr=lr_merged, epochs=epochs_merged)

      if device == "cuda": peak_ram_bytes = torch.cuda.max_memory_allocated(device)
      else: peak_ram_bytes = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
      end_time = time.time()

      acc_after_naive, f1_after_naive = evaluate_model(naive_merged_model, loader_full_test, device)
      results["Naive Average (No Align)"] = {
          'acc_zero_shot': acc_before_naive,
          'acc_finetuned': acc_after_naive,
          'f1_zero_shot': f1_before_naive,
          'f1_finetuned': f1_after_naive,
          'time': end_time - start_time,
          'ram_gb': peak_ram_bytes / 1024**3,
          'details': '100% dei layer fusi, 100% dei parametri addestrati.'
      }
  else:
      print("L'allineamento non ha modificato i pesi di model_B quindi testare il modello non allineato non ha senso.")

  for name, data in merged_models_to_evaluate.items():
      original_merged_model, trainable_indices = data["model"], data["indices"]
      
      for strategy in ["Only Head", "Selective FT", "Full FT"]:
          #se la fusione è totale, "Selective FT" è identico a "Full FT"
          if strategy == "Full FT" and ("Full" in name or "No Merge" in name):
              print(f"\n--- SALTO: {name} ({strategy}) perché ridondante con Selective FT ---")
              continue

          exp_name = f"{name} ({strategy})"
          print(f"\n--- Analisi: {exp_name} ---")
          
          model_to_eval = deepcopy(model)
          transfer_body_weights(original_merged_model, model_to_eval)
          model_to_eval.to(device)
          acc_zero_shot, f1_zero_shot = evaluate_model(model_to_eval, loader_full_test, device)
          print(f"Accuratezza Zero-Shot: {acc_zero_shot:.2f}%, F1-Score: {f1_zero_shot:.2f}%")

          if strategy == "Selective FT":
              freeze_layers_selectively(model_to_eval, trainable_indices)
              print("Strategia: Fine-tuning selettivo (layer fusi + testa).")
          elif strategy == "Only Head":
              freeze_layers_selectively(model_to_eval, []) # Congela tutto tranne la testa
              print("Strategia: Fine-tuning solo della testa (head).")
          else: # Full FT
              print("Strategia: Fine-tuning completo (nessun layer congelato).")
          
          start_time = time.time()
          if device == "cuda": torch.cuda.reset_peak_memory_stats(device)
          else: tracemalloc.start()

          train_model(model_to_eval, loader_full_train, loader_full_test, device, lr=lr_merged, epochs=epochs_merged)

          if device == "cuda": peak_ram_bytes = torch.cuda.max_memory_allocated(device)
          else: peak_ram_bytes = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
          end_time = time.time()
          acc_finetuned, f1_finetuned = evaluate_model(model_to_eval, loader_full_test, device)

          #Salvataggio dei risultati in un dizionario unificato
          results[exp_name] = {
              'acc_zero_shot': acc_zero_shot,
              'acc_finetuned': acc_finetuned,
              'f1_zero_shot': f1_zero_shot,
              'f1_finetuned': f1_finetuned,
              'time': end_time - start_time,
              'ram_gb': peak_ram_bytes / 1024**3
          }
  return results



