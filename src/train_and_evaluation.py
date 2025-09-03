import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """
    Funzione per addestrare il modello
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    print(f"Inizio addestramento per {epochs} epoche...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        accuracy, f1 = evaluate_model(model, test_loader)

        print(f"Epoch {epoch+1}/{epochs} -> "
              f"Loss: {epoch_loss:.4f}, "
              f"Accuracy sul Test Set: {accuracy:.2f}%, F1-Score: {f1:.2f}%")

    print("Addestramento completato.")

def evaluate_model(model, loader):
    """
    Funzione per valutare l'accuratezza di un modello usando come metriche
    l'accuratezza e il F1-Score.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    f1 = 100*f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1
