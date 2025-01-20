import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import PaperModel
from evaluate import visualize_grad_cam, visualize_latent_space, evaluate_model

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

# Validation/Test Dataset
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load the model weights
path = './weights/' + 'paper_model_without_amc_ep200.pth'
model_paper = PaperModel(num_classes=10)  # Reinitialize the model
model_paper.load_state_dict(torch.load(path, map_location=device))
model_paper = model_paper.to(device)
model_paper.eval()  # Set the model to evaluation mode

path = './weights/' + 'paper_model_with_amc_ang_1p25_ep200.pth'
model_paper_amc = PaperModel(num_classes=10)  # Reinitialize the model
model_paper_amc.load_state_dict(torch.load(path, map_location=device))
model_paper_amc = model_paper_amc.to(device)
model_paper_amc.eval()  # Set the model to evaluation mode
print("Model weights loaded and ready for inference.")


class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]
# Evaluate on Test Set
evaluate_model(model_paper, test_loader, device)
print('--- run visualization ---')
#visualize_latent_space(model_paper, test_loader, device, class_names)
visualize_grad_cam(model_paper, test_loader, device, class_names, 21)
visualize_grad_cam(model_paper_amc, test_loader, device, class_names, 21)