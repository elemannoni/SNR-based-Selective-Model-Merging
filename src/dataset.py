import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

class RemappedDataset(Dataset):
    """
    Dataset che rimappa le etichette originali in nuove etichette
    """
    def __init__(self, subset, original_indices):
        self.subset = subset
        self.mapping = {orig_label: new_label for new_label, orig_label in enumerate(original_indices)}

    def __getitem__(self, index):
        image, original_label = self.subset[index]
        remapped_label = self.mapping[original_label]
        return image, remapped_label

    def __len__(self):
        return len(self.subset)

def create_dataloader_CIFAR10(A_indices, B_indices, batch_size = 1024):
  """
  Funzione per scaricare CIFAR10 e creare i dataset e i dataloader
  """
  #Trasformazioni per normalizzare i dati
  transform_pipeline = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  #Caricamento dei dataset
  train_data_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_pipeline)
  test_data_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_pipeline)
  
  train_indices_A = [i for i, label in enumerate(train_data_full.targets) if label in A_indices]
  train_indices_B = [i for i, label in enumerate(train_data_full.targets) if label in B_indices]
  test_indices_A = [i for i, label in enumerate(test_data_full.targets) if label in A_indices]
  test_indices_B = [i for i, label in enumerate(test_data_full.targets) if label in B_indices]
  
  #Creazione dataset
  dataset_A = RemappedDataset(Subset(train_data_full, train_indices_A), A_indices)
  dataset_B = RemappedDataset(Subset(train_data_full, train_indices_B), B_indices)
  test_dataset_A = RemappedDataset(Subset(test_data_full, test_indices_A), A_indices)
  test_dataset_B = RemappedDataset(Subset(test_data_full, test_indices_B), B_indices)
  
  #Creazione dataloader
  loader_A = DataLoader(dataset_A, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
  loader_B = DataLoader(dataset_B, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
  test_loader_A = DataLoader(test_dataset_A, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
  test_loader_B = DataLoader(test_dataset_B, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
  loader_full_train = DataLoader(train_data_full, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
  loader_full_test = DataLoader(test_data_full, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
  
  
  print(f"Creato dataset di training A con {len(dataset_A)} campioni.")
  print(f"Creato dataset di training B con {len(dataset_B)} campioni.")
  print(f"Creato dataset di test A con {len(test_dataset_A)} campioni.")
  print(f"Creato dataset di test B con {len(test_dataset_B)} campioni.")
  return A_indices, B_indices, loader_A, loader_B, test_loader_A, test_loader_B, loader_full_train, loader_full_test
