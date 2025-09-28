# Medical AI Assistant - Federated Learning Demo

A web application that demonstrates the power of federated learning for medical image classification. This service uses models trained through federated learning across multiple virtual clinics to provide AI-assisted medical diagnosis.

## 🏥 Features

- **Multi-Domain Medical Classification**: Supports 5 different medical imaging domains
- **Privacy-Preserving AI**: Built on federated learning principles
- **Real-time Inference**: Fast image classification with confidence scores
- **Modern Web Interface**: Responsive design with drag-and-drop upload
- **Medical Domains Supported**:
  - **Pathological Images**: Lung tissue classification
  - **Dermatological Images**: Skin lesion analysis
  - **Retinal Images**: Eye disease detection
  - **Blood Cell Images**: Blood cell type classification
  - **Organ Images**: CT scan organ identification

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Trained federated learning model (from the Flower hackathon)

### Installation

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Model**:

   - Train your federated learning model using the enhanced Flower app
   - Place the trained model as `models/final_model.pt`

3. **Run the Service**:

   ```bash
   python app.py
   ```

4. **Access the Application**:
   - Open your browser to `http://localhost:5000`
   - Upload medical images for AI-powered classification

## 🧠 Technical Architecture

### Enhanced Model Features

- **Improved CNN Architecture**: 4-layer CNN with batch normalization and dropout
- **Data Augmentation**: Random flips, rotations, color jitter, and affine transforms
- **Advanced Training**: AdamW optimizer with cosine annealing and gradient clipping
- **FedProx Strategy**: Better convergence in federated settings

### Web Service Features

- **Flask Backend**: RESTful API for image classification
- **Real-time Processing**: Fast inference with PyTorch
- **Multi-format Support**: Handles various image formats and sizes
- **Confidence Scoring**: Provides probability distributions for predictions

## 📊 Model Performance Improvements

### Original vs Enhanced Model

| Aspect            | Original      | Enhanced               |
| ----------------- | ------------- | ---------------------- |
| Architecture      | 2 conv + 3 FC | 4 conv + 3 FC with BN  |
| Training Epochs   | 1             | 3                      |
| Server Rounds     | 3             | 10                     |
| Client Fraction   | 0.5           | 0.8                    |
| Learning Rate     | 0.01          | 0.001                  |
| Strategy          | FedAvg        | FedProx                |
| Data Augmentation | None          | Comprehensive          |
| Regularization    | None          | Dropout + Weight Decay |

### Expected Performance Gains

- **Accuracy**: 15-25% improvement across all datasets
- **Convergence**: Faster and more stable training
- **Generalization**: Better performance on unseen data
- **Robustness**: More reliable predictions

## 🏗️ Deployment Options

### Local Development

```bash
python app.py
```

### Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t medical-ai-service .
docker run -p 5000:5000 medical-ai-service
```

### Cloud Deployment

- **AWS**: Deploy on EC2 with ECS or Lambda
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Deploy on Container Instances or App Service

## 🔒 Privacy & Security

- **Federated Learning**: Models trained without sharing raw data
- **Local Processing**: Images processed locally, not stored
- **Secure Communication**: HTTPS recommended for production
- **Data Minimization**: Only necessary data is processed

## 🎯 Use Cases

### Healthcare Providers

- **Primary Care**: Quick screening for common conditions
- **Specialist Referral**: Triage patients to appropriate specialists
- **Rural Healthcare**: Bring AI expertise to underserved areas

### Medical Education

- **Training Tool**: Help medical students learn image interpretation
- **Case Studies**: Interactive learning with real-world examples

### Research

- **Data Collection**: Gather anonymized insights from diverse populations
- **Model Validation**: Test federated learning approaches

## 🚨 Important Disclaimers

- **Not a Replacement**: This tool is for assistance only, not diagnosis
- **Professional Consultation**: Always consult qualified medical professionals
- **Regulatory Compliance**: Ensure compliance with local medical device regulations
- **Data Privacy**: Follow HIPAA and other privacy regulations

## 🔧 Configuration

### Model Configuration

```python
# In app.py, modify these parameters:
model_path = "models/final_model.pt"
num_classes = 9  # Adjust based on dataset
dataset = "pathmnist"  # Default dataset
```

### Web Service Configuration

```python
# Flask configuration
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📈 Monitoring & Analytics

- **Health Check**: `/api/health` endpoint for service monitoring
- **Performance Metrics**: Track inference time and accuracy
- **Usage Analytics**: Monitor service usage patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Flower Framework**: For federated learning infrastructure
- **MedMNIST Dataset**: For medical imaging data
- **Flask Community**: For web framework
- **PyTorch Team**: For deep learning framework

## 📞 Support

For questions or support, please:

- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with ❤️ for the Flower Decentralized AI Hackathon**
