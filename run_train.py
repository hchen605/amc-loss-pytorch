import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import train_model, AMCLoss, AMCLossWithPairing
from evaluate import evaluate_model
from model import PaperModel


# Check device compatibility (including Apple Silicon MPS support)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Validation/Test Dataset
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Validation/Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(' --- Data loaded ---', flush=True)

# Initialize model, optimizer, and AMC-Loss
model_paper = PaperModel()
optimizer = torch.optim.Adam(model_paper.parameters(), lr=0.001)

# # Train without AMC-Loss
# print("Training without AMC-Loss", flush=True)
# num_epochs = 300
# train_model(model_paper, criterion=F.cross_entropy, optimizer=optimizer, dataloader=train_loader, num_epochs=num_epochs, use_amc_loss=False)

# # Save the model weights
# path = './weights/' +  'paper_model_without_amc_' + 'ep' + str(num_epochs) + '.pth'
# torch.save(model_paper.state_dict(), path)
# print("Model weights saved")

# Evaluate on Test Set
# evaluate_model(model_paper, test_loader, device)

# Train with AMC-Loss
print("Training with AMC-Loss")
num_epochs = 200
criterion_amc = AMCLossWithPairing(angular_margin=1, lambda_=0.1)
model_paper_amc = PaperModel()  # Reinitialize the model
optimizer = torch.optim.Adam(model_paper_amc.parameters(), lr=0.001)
train_model(model_paper_amc, criterion=criterion_amc, optimizer=optimizer, dataloader=train_loader, num_epochs=num_epochs, use_amc_loss=True)

path = './weights/' +  'paper_model_with_amc_ang_1_lambda_0p05' + 'ep' + str(num_epochs) + '.pth'
torch.save(model_paper_amc.state_dict(), path)
print("Model weights saved")
# Evaluate on Test Set
evaluate_model(model_paper_amc, test_loader, device)