import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os

transform = transforms.ToTensor()

mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

x_train = [torch.flatten(image) for image, _ in mnist_train]
y_train = [label for _, label in mnist_train]

x_test = [torch.flatten(image) for image, _ in mnist_test]
y_test = [label for _, label in mnist_test]

x = torch.stack(x_train).float()
y = torch.tensor(y_train, dtype=torch.long)
x_test = torch.stack(x_test).float()
y_test = torch.tensor(y_test, dtype=torch.long)

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 10)
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = DigitNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
losses = []

if os.path.exists("digit_net.pth"):
    model.load_state_dict(torch.load("digit_net.pth"))
    model.eval()
    print("Model loaded from saved file. Skipping training.")
else:
    print("No saved model found. Starting training from scratch...")
    for epoch in range(2000):
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "digit_net.pth")
    print("Model saved to digit_net.pth")

with torch.no_grad():
    preds = model(x_test)
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

index =456
img = x_test[index].unsqueeze(0)
output = model(img)
pred = torch.argmax(output, dim=1).item()
print("Prediction:", pred)

plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {pred}, Actual: {y_test[index].item()}")
plt.axis('off')
plt.show()

if losses:
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
