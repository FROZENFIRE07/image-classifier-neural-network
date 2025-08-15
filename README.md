# Image Classifier Neural Network  

A PyTorch-based neural network that classifies handwritten digits from the MNIST dataset.  
Built as part of my practice and experimentation to strengthen deep learning fundamentals.  

## 📌 Features  
- Dataset: MNIST (28x28 grayscale images)  
- Architecture: 784 → 16 → 16 → 10  
- Activation: ReLU  
- Loss: CrossEntropyLoss  
- Optimizer: Adam (lr = 0.004)  
- Saves & loads model with `.pth`  
- Visualizes training loss with Matplotlib  
- Predicts and displays single digit  

## 📂 Project Structure

```plaintext
├── data/             # MNIST dataset
├── digit_net.pth     # Saved model
├── main.py           # Training & testing
└── README.md         # Project documentation
```


## 📊 Output
Console: Test Accuracy
Matplotlib: Training loss curve
Image: Predicted vs Actual digit

## 🧠 Purpose

This is a practice project for learning, experimenting, and improving my deep learning skills using PyTorch.

## 🚀 Run the Project  
```bash
git clone https://github.com/FROZENFIRE07/image-classifier-neural-network.git
cd image-classifier-neural-network
pip install torch torchvision matplotlib
python main.py 

