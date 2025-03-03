import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model import Conv1DModel
from evaluate import visualize_grad_cam_ecg, visualize_latent_space, evaluate_model, compute_infidelity_ecg
import argparse
from train import collate_fn
import torchaudio
import torchaudio.transforms as T
import os
import numpy as np

# Check device compatibility (including Apple Silicon MPS support)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECG5000Dataset(Dataset):
    """
    ECG5000 dataset loader.
    Assumes data files 'ECG5000_TRAIN.txt' and 'ECG5000_TEST.txt' are located in data_path.
    Each row should have the label as the first element (1-indexed) and the time-series values following.
    """
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        super().__init__()
        self.transform = transform

        if split.lower() == 'train':
            file_path = os.path.join(data_path, "ECG5000_TRAIN.txt")
        elif split.lower() == 'test':
            file_path = os.path.join(data_path, "ECG5000_TEST.txt")
        else:
            raise ValueError("split must be 'train' or 'test'")

        # Load data; each row: [label, x1, x2, ...]
        data = np.loadtxt(file_path)
        # The first column is the label; convert to 0-indexed
        self.labels = (data[:, 0] - 1).astype(np.int64)
        # The remaining columns are the time-series values
        self.data = data[:, 1:]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert the sample to a torch tensor and add a channel dimension (for 1D CNNs)
        sample = torch.tensor(self.data[idx], dtype=torch.float).unsqueeze(0)
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


parser = argparse.ArgumentParser(description="Train a model with AMC-Loss on the ECG5000 dataset.")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs (unused if only training with AMC-Loss)')
parser.add_argument('--num_epochs_amc', type=int, default=50, help='Number of epochs for training with AMC-Loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--angular', type=float, default=1, help='AMC-loss angular margin (default: 1)')
parser.add_argument('--lambda_', type=float, default=0.1, help='AMC-loss lambda (default: 0.1)')
parser.add_argument('--save_path', type=str, default='./weights', help='Path to save the model weights')
parser.add_argument('--data_path', type=str, default='./data_ecg5000', help='Path to ECG5000 data')
args = parser.parse_args()


test_dataset = ECG5000Dataset(data_path=args.data_path, split='test')

num_classes = 5
input_channels = 1

print('--- ECG5000 Data Loaded ---')


# -------------------------------
# Create DataLoaders
# -------------------------------
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)


print(' --- Data loaded ---', flush=True)
# Load the model weights
# Initialize model, optimizer, and AMC-Loss

class_names_ecg5000 = [
    "Normal",         # Normal beat
    "LBBB",           # Left Bundle Branch Block
    "RBBB",           # Right Bundle Branch Block
    "PVC",            # Premature Ventricular Contraction
    "APC"             # Premature Atrial Contraction
]

path = './weights/' + 'ecg/conv1d_model_without_amc.pth'
model_paper = Conv1DModel(input_channels=input_channels, num_classes=num_classes)  # Reinitialize the model
model_paper.load_state_dict(torch.load(path, map_location=device))
model_paper = model_paper.to(device)
model_paper.eval()  # Set the model to evaluation mode

path = './weights/' + 'ecg/conv1d_model_with_amcpair_ang_0.5_lambda_0.2.pth'
model_paper_amc = Conv1DModel(input_channels=input_channels, num_classes=num_classes) 
model_paper_amc.load_state_dict(torch.load(path, map_location=device))
model_paper_amc = model_paper_amc.to(device)
model_paper_amc.eval()  # Set the model to evaluation mode
print("Model weights loaded and ready for inference.")



# Evaluate on Test Set
#evaluate_model(model_paper, test_loader, device)
#evaluate_model(model_paper_amc, test_loader, device)
#print('--- run visualization ---')
#visualize_latent_space(model_paper, test_loader, device, class_names_ecg5000, 'ecg')
#visualize_latent_space(model_paper_amc, test_loader, device, class_names_ecg5000, 'ecg', use_amc_loss=True)
idx = 926
visualize_grad_cam_ecg(model_paper, test_loader, device, class_names_ecg5000, idx)
visualize_grad_cam_ecg(model_paper_amc, test_loader, device, class_names_ecg5000, idx, use_amc_loss=True)

#print('--- run Infidelity ---')
# Compute infidelity for the test dataset
#infidelity_score = compute_infidelity_ecg(model_paper, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score: ', infidelity_score)
#infidelity_score = compute_infidelity_ecg(model_paper_amc, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score (AMC): ', infidelity_score)
#infidelity_score = compute_infidelity_sample(model_paper, test_loader, device, idx, noise_std=0.003, perturbations=50)
#infidelity_score = compute_infidelity_sample(model_paper_amc, test_loader, device, idx, noise_std=0.003, perturbations=50)

