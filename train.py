import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import yaml
import json

# DEVICE ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# PARAMS ---------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

epochs = params["epochs"]
lr = float(params["lr"])

# DATASET ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
#разделение_данных
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(
    dataset,
    [train_size, val_size, test_size]
)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# MODEL ------------
model = models.resnet18(pretrained = True)
model.fc = nn.Linear(in_features=512, out_features=6)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# TRAIN ------------------------
train_losses = []

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(f"Epoch Loss: {epoch_loss:.4f}")

# TEST ------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:}")

torch.save(model.state_dict(), "best_model.pth")

with open("metrics.json", "w") as f:
    json.dump({"test_accuracy": test_accuracy}, f)

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_plot.png")