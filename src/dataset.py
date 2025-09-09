import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image, ImageFilter

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class RandomGaussianBlur(object):
    def __call__(self, img, start = 1, end = 3):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(start, end)))
        
def create_augmented_dataloaders_CIFAR10(batch_size=1024, num_workers=8):
    # transform_geometric = transforms.Compose([
    #     transforms.RandomGrayscale(p=1.0),   # forza grigio = forma
    #     transforms.ColorJitter(0.8, 0.8, 0.8, 0.5),  # colori "disturbanti"
    #     transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), shear=20),
    #     transforms.ToTensor(),
    #     AddGaussianNoise(0., 0.15),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                 (0.2023, 0.1994, 0.2010)),
    # ])
    
    # transform_photometric = transforms.Compose([
    #     transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    #     transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), shear=25),  # forme deformate
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Lambda(lambda img: RandomGaussianBlur()(img)),  # sfocatura forte
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # patch random
    #     AddGaussianNoise(0., 0.2),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                 (0.2023, 0.1994, 0.2010)),
    # ])

    transform_geometric = transforms.Compose([
        # La scala di grigi ora è un'opzione, non una certezza
        transforms.RandomGrayscale(p=0.3),
        # Jitter di colore meno intenso
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        # Deformazioni geometriche ridotte
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        transforms.ToTensor(),
        # Rumore Gaussiano diminuito
        AddGaussianNoise(0., 0.05),
        # Normalizzazione standard di CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_photometric = transforms.Compose([
        # Il crop casuale ora è meno aggressivo
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        # Deformazioni geometriche molto più contenute
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), shear=15),
        transforms.RandomHorizontalFlip(),
        # Sfocatura standard con parametri moderati
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # L'occlusione casuale rimane un buon test
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        # Rumore Gaussiano diminuito
        AddGaussianNoise(0., 0.08),
        # Normalizzazione standard di CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    # (Le altre pipeline rimangono invariate per i test e il fine-tuning)
    transform_structural_hard = transforms.Compose([
      transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      transforms.RandomErasing(p=0.7, scale=(0.05, 0.25), ratio=(0.3, 3.3)),
      AddGaussianNoise(0., 0.1) 
    ])
    transform_standard = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_extreme_test = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),              # meno aggressivo
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=5),  
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),                      # piccoli cambi colore
        transforms.RandomGrayscale(p=0.2),                               # meno frequente
        transforms.Lambda(lambda img: RandomGaussianBlur()(img, start = 0.3, end = 1.0)),        # blur molto leggero
        transforms.RandomHorizontalFlip(p=0.3),                          # meno probabile
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.5, 2.0)),  # piccole occlusioni
        AddGaussianNoise(0., 0.05),                                      # rumore leggero
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data_A = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_geometric)
    train_data_B = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_photometric)
    train_data_C = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_structural_hard)
    train_data_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_standard)
    test_data_standard = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_standard)
    test_data_hard = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_extreme_test)
    test_data_C = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_structural_hard)
    
    loader_A_train = DataLoader(train_data_A, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    loader_B_train = DataLoader(train_data_B, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    loader_C_train = DataLoader(train_data_C, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    loader_C_test = DataLoader(test_data_C, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    loader_full_train = DataLoader(train_data_full, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    loader_full_test = DataLoader(test_data_standard, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    loader_full_test_hard = DataLoader(test_data_hard, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    print("--- Dataloader Creati con Test Set Estremo ---")
    return (loader_A_train, loader_B_train, loader_C_train, loader_C_test,
          loader_full_train, loader_full_test, loader_full_test_hard)