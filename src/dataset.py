import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def create_augmented_dataloaders_CIFAR10(batch_size=1024, num_workers=8):
  """
  Crea un set completo di dataloader per esperimenti di model merging e fine-tuning.
  """
  
  transform_geometric = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])


  transform_photometric = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomPosterize(bits=4, p=0.4),
    transforms.RandomEqualize(p=0.4),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddGaussianNoise(0., 0.08)
  ])


  transform_structural = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomSolarize(threshold=128, p=0.3),
      transforms.RandomAutocontrast(p=0.3),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
  ])


  transform_standard = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  transform_hard_test = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    AddGaussianNoise(0., 0.05)
  ])

  train_data_A = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_geometric)
  train_data_B = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_photometric)
  train_data_C = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_structural)
  train_data_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_standard)
  train_data_full_hard = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_hard_test)

  test_data_standard = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_standard)
  test_data_hard = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_hard_test)
  test_data_C = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_structural)

  loader_A = DataLoader(train_data_A, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  loader_B = DataLoader(train_data_B, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  loader_C_train = DataLoader(train_data_C, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  
  loader_full_train = DataLoader(train_data_full, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  loader_full_train_hard = DataLoader(train_data_full_hard, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

  
  loader_full_test = DataLoader(test_data_standard, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
  loader_full_test_hard = DataLoader(test_data_hard, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
  loader_C_test = DataLoader(test_data_C, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
  
  print("--- Dataloader Creati ---")
  print(f"Loader A (Geometric): {len(train_data_A)} campioni.")
  print(f"Loader B (Photometric): {len(train_data_B)} campioni.")
  print(f"Loader C (Structural/Occlusion): {len(train_data_C)} campioni per train, {len(test_data_C)} per test.")
  print(f"Test Loader Standard: {len(test_data_standard)} campioni.")
  print(f"Test Loader Difficile: {len(test_data_hard)} campioni.")

  return loader_A, loader_B, loader_C_train, loader_C_test, loader_full_train, loader_full_train_hard, loader_full_test, loader_full_test_hard