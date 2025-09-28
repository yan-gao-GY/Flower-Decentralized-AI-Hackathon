"""medapp: A Flower / pytorch_msg_api app."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, 
    Normalize, 
    ToTensor, 
    RandomHorizontalFlip, 
    RandomRotation, 
    ColorJitter,
    RandomAffine
)


class Net(nn.Module):
    """Enhanced CNN model for medical image classification"""

    def __init__(self, num_classes: int, input_channels: int = 3):
        super(Net, self).__init__()
        
        # Enhanced convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: 64x64, after 4 pooling operations: 64/16 = 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def train(net, trainloader, epochs, lr,device):
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight = decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler.step()
    running_loss = 0.0 
    total_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            total_batches += 1
        scheduler.step()
        running_loss += epoch_loss
    avg_trainloss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_trainloss

# Enhanced transforms with data augmentation for training
train_transforms = Compose([
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Basic transforms for validation/test
val_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Default to val_transforms for backward compatibility
pytorch_transforms = val_transforms


def apply_train_transforms(batch):
    """Apply training transforms to the batch."""
    batch["image"] = [train_transforms(img) for img in batch["image"]]
    return batch

def apply_val_transforms(batch):
    """Apply validation transforms to the batch."""
    batch["image"] = [val_transforms(img) for img in batch["image"]]
    return batch

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def apply_transforms(batch):
    batch["partition"] = load_from_disk(batch["partition"])
    return batch



def load_data(data_path: str):
    """Load partition with separate train/val transforms."""
    partition = load_from_disk(data_path)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Apply different transforms for train and test
    train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
    test_partition = partition_train_test["test"].with_transform(apply_val_transforms)
    
    # Construct dataloaders with larger batch size for better training
    trainloader = DataLoader(train_partition, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test_partition, batch_size=64, num_workers=2)
    return trainloader, testloader

def predict_image(image: Image.Image, dataset: str) -> Dict:
    """Make prediction on uploaded image"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        input_tensor = preprocess_image(image, dataset)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

def predict_image(image: Image.Image, dataset: str) -> Dict:
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        input_tensor = preprocess_image(image, dataset)
    
    month.eval()
    outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probabilities, 1)

    class_info = class_mappings.get(dataset)
    class_names= class_info.get('classes', [f'Class {i}' for i in range(len(probabilities[0]))])
    top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
    results = {
        'predicted_class': class_names[predicted_class.item()],
        'confidence': confidence.item(),
        'top_predictions': [
            {
                'class': class_names[idx.item()],
                'probability': prob.item()
            }
            for prob, idx in zip(top3_probs[0], top3_indices[0])
        ]
    }

    return results
    except Exception as e:
        print(f"error in predict_image: {e}")
        return {"error": str(e)}

    return results
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return {"error": str(e)}

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        input_tensor = preprocess_image(image, dataset)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        class_info = class_mappings.get(dataset, {})
        class_names = class_info.get('classes', [f'Class {i}' for i in range(len(probabilities[0]))])
        
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
        
        results = {
            'predicted_class': class_names[predicted_class.item()],
            'confidence': confidence.item(),
            'top_predictions': [
                {
                    'class': class_names[idx.item()],
                    'probability': prob.item()
                }
                for prob, idx in zip(top3_probs[0], top3_indices[0])
            ]
        }
        return results

    except Exception as e:
        print(f"Error in test: {e}")
        return {"error": str(e)}

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set with enhanced optimization."""
    net.to(device)  # move model to GPU if available
    net.train()
    
    # Enhanced loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    running_loss = 0.0
    total_batches = 0
    


    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            total_batches += 1
        
        # Update learning rate
        scheduler.step()
        running_loss += epoch_loss
    
    avg_trainloss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy



def maybe_init_wandb(use_wandb: bool, wandbtoken: str) -> None:
    """Initialize Weights & Biases if specified in run_config."""
    if use_wandb:
        if not wandbtoken:
            print(
                "W&B token wasn't found. Set it by passing `--run-config=\"wandb-token='<YOUR-TOKEN>'\" to your `flwr run` command.",
            )
            use_wandb = False
        else:
            os.environ["WANDB_API_KEY"] = wandbtoken
            wandb.init(project="Flower-hackathon-MedApp")


def load_centralized_dataset(data_path: str):
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_from_disk(data_path)
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        input_tensor = preprocess_image(image, dataset)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        correct += (predicted_)

        class_info = class_mappings.get(dataset, {})
        class_names = class_info.get('classes', [f'Class {i}' for i in range(len(probabilities[0]))])
        
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
        