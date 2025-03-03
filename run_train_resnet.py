import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ssl
import argparse
from train import train_model_resnet, AMCLossWithPairing
from evaluate import evaluate_model
from model import PaperModel, get_model, define_tsnet  # Your custom model
import torch.nn as nn

# Disable SSL verification for dataset downloading
ssl._create_default_https_context = ssl._create_unverified_context

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with optional AMC-Loss on various datasets.")

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                    help='Dataset to use (default: cifar10)')

parser.add_argument('--model_type', type=str, default='paper_model',
                    choices=['paper_model', 'resnet18', 'resnet50'],
                    help='Model to train (default: paper_model)')

parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained weights for ResNet (ignored for paper_model).')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training (default: 128)')

parser.add_argument('--num_epochs', type=int, default=200,
                    help='Number of epochs for training without AMC-Loss (default: 200)')

parser.add_argument('--num_epochs_amc', type=int, default=200,
                    help='Number of epochs for training with AMC-Loss (default: 200)')

parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning rate for optimizer (default: 0.001)')

parser.add_argument('--angular', type=float, default=1,
                    help='AMC-loss angular (default: 1)')

parser.add_argument('--lambda_', type=float, default=0.1,
                    help='AMC-loss angular (default: 0.1)')

parser.add_argument('--save_path', type=str, default='./weights/cifar10/',
                    help='Path to save the model weights (default: ./weights/cifar10/)')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 regularization) for the optimizer (default: 1e-4)')

parser.add_argument('--scheduler_step', type=int, default=80,
                    help='Step size for the learning rate scheduler (default: 80)')

parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                    help='Gamma factor for the learning rate scheduler (default: 0.1)')

parser.add_argument('--note', type=str, default='sgd',
                    help='Note to save the model weights (default: ./weights/cifar10/)')

args = parser.parse_args()

# Check device compatibility
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset selection
if args.dataset == 'cifar10':
    # CIFAR10 has 3 input channels, 10 classes
    input_channels = 3
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

elif args.dataset == 'mnist':
    # MNIST has 1 input channel, 10 classes
    input_channels = 1
    num_classes = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print('--- Data loaded ---', flush=True)

# --------------------------
# 1) Train WITHOUT AMC-Loss
# --------------------------
# print("Training without AMC-Loss", flush=True)

# #model = get_model(args.model_type, input_channels, num_classes, pretrained=args.pretrained)
# model = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
# model.to(device)

# #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
# # Add a learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, 
#     step_size=args.scheduler_step, 
#     gamma=args.scheduler_gamma
# )

# # Train model
# save_name_no_amc = f"{args.model_type}_{args.note}_without_amc_ep_{args.num_epochs}.pth"
# train_model_resnet(model=model,
#             path=args.save_path + f"{args.model_type}__{args.note}_without_amc", 
#             criterion=F.cross_entropy,
#             optimizer=optimizer,
#             dataloader=train_loader,
#             dataloader_val=test_loader,
#             num_epochs=args.num_epochs,
#             use_amc_loss=False,
#             scheduler=scheduler
#             )

# #Save the model weights
# torch.save(model.state_dict(), args.save_path + save_name_no_amc)
# print(f"Model weights saved at {args.save_path + save_name_no_amc}")

# #Evaluate on Test Set
# evaluate_model(model, test_loader, device)

# ------------------------
# 2) Train WITH AMC-Loss
# ------------------------
print("Training with AMC-Loss", flush=True)

# Hyperparameters for AMC-Loss
angular_margin = args.angular
lambda_ = args.lambda_
criterion_amc = AMCLossWithPairing(angular_margin=angular_margin, lambda_=lambda_)

#model_amc = get_model(args.model_type, input_channels, num_classes, pretrained=args.pretrained)
model_amc = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
model_amc.to(device)

#optimizer_amc = torch.optim.Adam(model_amc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_amc = torch.optim.SGD(model_amc.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer_amc, 
    step_size=args.scheduler_step, 
    gamma=args.scheduler_gamma
)

save_name_amc = f"{args.model_type}_{args.note}_with_amcpair_ang_{angular_margin}_lambda_{lambda_}_ep{args.num_epochs_amc}.pth"
train_model_resnet(model=model_amc,
            path=args.save_path + f"{args.model_type}_{args.note}_with_amcpair_ang_{angular_margin}_lambda_{lambda_}",
            criterion=criterion_amc,
            optimizer=optimizer_amc,
            dataloader=train_loader,
            dataloader_val=test_loader,
            num_epochs=args.num_epochs_amc,
            use_amc_loss=True,
            scheduler=scheduler
            )

# Save the model weights
torch.save(model_amc.state_dict(), args.save_path + save_name_amc)
print(f"Model weights saved at {args.save_path + save_name_amc}")

# Evaluate on Test Set
evaluate_model(model_amc, test_loader, device)
