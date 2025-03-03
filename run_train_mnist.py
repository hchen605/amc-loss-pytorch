import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import train_model, AMCLossWithPairing
from evaluate import evaluate_model
from model import PaperModel, SimpleCNN
import ssl
import argparse

# Disable SSL verification for dataset downloading
ssl._create_default_https_context = ssl._create_unverified_context

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with optional AMC-Loss on various datasets.")

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                    help='Dataset to use (default: cifar10)')
parser.add_argument('--model_type', type=str, default='paper_model',
                    choices=['paper_model', 'simple_model'],
                    help='Model to train (default: paper_model)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training without AMC-Loss (default: 300)')
parser.add_argument('--num_epochs_amc', type=int, default=100, help='Number of epochs for training with AMC-Loss (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer (default: 0.001)')
parser.add_argument('--angular', type=float, default=1, help='AMC-loss angular (default: 1)')
parser.add_argument('--lambda_', type=float, default=0.1,help='AMC-loss angular (default: 0.1)')
parser.add_argument('--save_path', type=str, default='./weights/cifar10/', help='Path to save the model weights (default: ./weights/)')
parser.add_argument('--note', type=str, default='sgd', help='Note to save the model weights (default: ./weights/cifar10/)')
args = parser.parse_args()

# Check device compatibility
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset selection
if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print(' --- Data loaded ---', flush=True)

# Initialize model, optimizer, and AMC-Loss
if args.dataset == 'cifar10':
    input_channels = 3
elif args.dataset == 'mnist':
    input_channels = 1

if args.model_type == 'paper_model':
    model = PaperModel(input_channels=input_channels)
else:
    model = SimpleCNN(input_channels=input_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train without AMC-Loss
print("Training without AMC-Loss", flush=True)
path = args.save_path + f'{args.model_type}_without_amc'
train_model(model, path=path, criterion=F.cross_entropy, optimizer=optimizer, dataloader=train_loader, dataloader_val=test_loader,
            num_epochs=args.num_epochs, use_amc_loss=False)

#Save the model weights
path = args.save_path + f'{args.model_type}_without_amc_ep{args.num_epochs}_{args.note}.pth'
torch.save(model.state_dict(), path)
print(f"Model weights saved at {path}")

# Evaluate on Test Set
#evaluate_model(model_paper, test_loader, device)

# Train with AMC-Loss
print("Training with AMC-Loss")
angular_margin = args.angular
lambda_ = args.lambda_
criterion_amc = AMCLossWithPairing(angular_margin=angular_margin, lambda_=lambda_)

if args.model_type == 'paper_model':
    model_amc = PaperModel(input_channels=input_channels)  # Reinitialize the model
else:
    model_amc = SimpleCNN(input_channels=input_channels)
optimizer = torch.optim.Adam(model_amc.parameters(), lr=args.lr)
path = args.save_path + f'{args.model_type}_with_amcpair_ang_{angular_margin}_lambda_{lambda_}'
train_model(model_amc, path=path, criterion=criterion_amc, optimizer=optimizer, dataloader=train_loader, dataloader_val=test_loader,
            num_epochs=args.num_epochs_amc, use_amc_loss=True)

# Save the model weights
path = args.save_path + f'{args.model_type}_with_amcpair_ang_{angular_margin}_lambda_{lambda_}_{args.note}.pth'
torch.save(model_amc.state_dict(), path)
print(f"Model weights saved at {path}")

# Evaluate on Test Set
evaluate_model(model_amc, test_loader, device)
