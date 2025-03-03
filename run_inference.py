import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import PaperModel, get_model, define_tsnet
from evaluate import visualize_grad_cam, visualize_latent_space, evaluate_model, compute_infidelity, compute_infidelity_sample, visualize_feature_ablation, compute_feature_ablation
import argparse

# Check device compatibility (including Apple Silicon MPS support)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with optional AMC-Loss on various datasets.")

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                    help='Dataset to use (default: cifar10)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
parser.add_argument('--model_type', type=str, default='paper_model',
                    choices=['paper_model', 'resnet18', 'resnet50'],
                    help='Model to train (default: paper_model)')
parser.add_argument('--save_path', type=str, default='./weights/cifar10/', help='Path to save the model weights (default: ./weights/)')
args = parser.parse_args()

# Dataset selection
if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print(' --- Data loaded ---', flush=True)
# Load the model weights
# Initialize model, optimizer, and AMC-Loss

class_names_cifar10 = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

class_names_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

if args.dataset == 'cifar10':
    input_channels = 3
    class_names = class_names_cifar10
elif args.dataset == 'mnist':
    input_channels = 1
    class_names = class_names_mnist
num_classes = 10

path = './weights/' + 'cifar_re/paper_model_without_amc_ep300_re.pth'
model_paper = PaperModel(input_channels=input_channels)  # Reinitialize the model
#model_paper = get_model('resnet18', input_channels, 10, False)
#model_paper = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
model_paper.load_state_dict(torch.load(path, map_location=device))
model_paper = model_paper.to(device)
model_paper.eval()  # Set the model to evaluation mode

path = './weights/' + 'cifar_re/paper_model_with_amcpair_ang_0.5_lambda_0.1_re.pth'
model_paper_amc = PaperModel(input_channels=input_channels) 
#model_paper_amc = get_model('resnet18', input_channels, 10, False)  # Reinitialize the model
#model_paper_amc = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
model_paper_amc.load_state_dict(torch.load(path, map_location=device))
model_paper_amc = model_paper_amc.to(device)
model_paper_amc.eval()  # Set the model to evaluation mode
print("Model weights loaded and ready for inference.")



# Evaluate on Test Set
#evaluate_model(model_paper, test_loader, device)
#evaluate_model(model_paper_amc, test_loader, device)
#print('--- run visualization ---')
#visualize_latent_space(model_paper, test_loader, device, class_names, args.dataset)
#visualize_latent_space(model_paper_amc, test_loader, device, class_names, args.dataset, use_amc_loss=True)
idx = 21
visualize_grad_cam(model_paper, test_loader, device, class_names, idx, args.dataset)
visualize_grad_cam(model_paper_amc, test_loader, device, class_names, idx, args.dataset, use_amc_loss=True)
#visualize_feature_ablation(model_paper, test_loader, device, idx)
#visualize_feature_ablation(model_paper_amc, test_loader, device, idx, use_amc_loss=True)
# compute_feature_ablation(model_paper, test_loader, device, use_amc_loss=False)
# compute_feature_ablation(model_paper_amc, test_loader, device, use_amc_loss=True)

#print('--- run Infidelity ---')
# Compute infidelity for the test dataset
#infidelity_score = compute_infidelity(model_paper, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score: ', infidelity_score)
#infidelity_score = compute_infidelity(model_paper_amc, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score (AMC): ', infidelity_score)
#infidelity_score = compute_infidelity_sample(model_paper, test_loader, device, idx, noise_std=0.003, perturbations=50)
#infidelity_score = compute_infidelity_sample(model_paper_amc, test_loader, device, idx, noise_std=0.003, perturbations=50)

