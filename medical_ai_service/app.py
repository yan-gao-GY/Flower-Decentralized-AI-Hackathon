"""
Medical AI Service - A web application for medical image classification
Built on top of the federated learning model from the Flower hackathon
"""

import os
import io
import base64
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import json

# Import our enhanced model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medapp.task import Net

app = Flask(__name__)

# Global model and class mappings
model = None
class_mappings = {
    'pathmnist': {
        'name': 'Pathological Image Classification',
        'classes': ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma'],
        'description': 'Classifies pathological lung tissue images'
    },
    'dermamnist': {
        'name': 'Dermatological Image Classification', 
        'classes': ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesion'],
        'description': 'Classifies dermatological skin lesion images'
    },
    'retinamnist': {
        'name': 'Retinal Image Classification',
        'classes': ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD'],
        'description': 'Classifies retinal fundus images for eye diseases'
    },
    'bloodmnist': {
        'name': 'Blood Cell Classification',
        'classes': ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature granulocytes', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet'],
        'description': 'Classifies different types of blood cells'
    },
    'organamnist': {
        'name': 'Organ Image Classification',
        'classes': ['Spleen', 'Right Kidney', 'Left Kidney', 'Gallbladder', 'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland'],
        'description': 'Classifies organ images from CT scans'
    }
}

def load_model(model_path: str, num_classes: int, dataset: str):
    """Load the trained federated learning model"""
    global model
    try:
        model = Net(num_classes=num_classes, input_channels=3 if dataset != 'organamnist' else 1)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            # Initialize with random weights if no model found
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).items()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy
    
def test_fn(net, testloader, device):
    return test(net, testloader, device)


def apply_train_transforms(batch):
    batch["image"] = [train_transforms(img) for img in batch["image"]]
    return batch

def apply_val_transforms(batch):
    batch["partition"] = load_from_disk(batch["partition"])
    return batch

def preprocess_image(image: Image.Image, dataset: str) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    # Resize to 64x64 as expected by the model
    image = image.resize((64, 64))
    
    # Convert to RGB if needed (except for organamnist which is grayscale)
    if dataset != 'organamnist':
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        if image.mode != 'L':
            image = image.convert('L')
    
    # Convert to tensor and normalize
    image_array = np.array(image)
    if dataset == 'organamnist':
        # Grayscale image
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    else:
        # RGB image
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels)
            correct += (torch.max(outputs.data, ))

    train_transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degree=10),
        ColorJItter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        RandomAffine(degrees = 0, translate=(0.1, 0.1), scale=(0.9, 1.1),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transformers = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pytorch_transforms = val_transforms


def load_data(data_path: str):
    partition = load_from_disk(data)

def predict_image(image: Image.Image, dataset: str) -> Dict:
    """Make prediction on uploaded image"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(image, dataset)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        # Get class names
        class_info = class_mappings.get(dataset, {})
        class_names = class_info.get('classes', [f'Class {i}' for i in range(len(probabilities[0]))])
        
        # Get top 3 predictions
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
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', datasets=class_mappings.keys())

@app.route('/classify', methods=['POST'])
def classify():
    """Classify uploaded image"""
    try:
        # Get dataset type
        dataset = request.form.get('dataset', 'pathmnist')
        
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"})
        
        # Load and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Make prediction
        result = predict_image(image, dataset)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"})

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets and their information"""
    return jsonify(class_mappings)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "available_datasets": list(class_mappings.keys())
    })

if __name__ == '__main__':
    # Load model on startup
    model_path = "models/final_model.pt"
    num_classes = 9  # Default for pathmnist
    dataset = "pathmnist"
    
    if load_model(model_path, num_classes, dataset):
        print("Model loaded successfully!")
    else:
        print("Failed to load model, running with random weights")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
