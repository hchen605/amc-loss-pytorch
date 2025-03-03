import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import PaperModel, get_model, define_tsnet
from evaluate import visualize_grad_cam_sc, visualize_latent_space, evaluate_model, compute_infidelity, compute_infidelity_sample
import argparse
from train import collate_fn
import torchaudio
import torchaudio.transforms as T
import os

# Check device compatibility (including Apple Silicon MPS support)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root: str, subset: str, transform=None, download: bool = True):
        super().__init__(root=root, download=download)
        self.transform = transform

        def load_list(filename):
            with open(filename) as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        self._val_list = "./data_speechcommands/SpeechCommands/speech_commands_v0.02/validation_list.txt"      
        self._test_list = "./data_speechcommands/SpeechCommands/speech_commands_v0.02/testing_list.txt"
        if subset == "validation":
            self._walker = load_list(self._val_list)
        elif subset == "testing":
            self._walker = load_list(self._test_list)
        elif subset == "training":
            excludes = load_list(self._val_list) + load_list(self._test_list)
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        else:
            raise ValueError("subset must be 'training', 'validation', or 'testing'")

    def __getitem__(self, n):
        waveform, sample_rate, label, *_ = super().__getitem__(n)
        if self.transform:
            # Apply transform (e.g. MelSpectrogram)
            waveform = self.transform(waveform)
        return waveform, label

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with optional AMC-Loss on various datasets.")

parser.add_argument('--dataset', type=str, default='speechcommands', choices=['cifar10', 'mnist', 'speechcommands'],
                    help='Dataset to use (default: cifar10)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
parser.add_argument('--model_type', type=str, default='paper_model',
                    choices=['paper_model', 'resnet18', 'resnet50'],
                    help='Model to train (default: paper_model)')
parser.add_argument('--save_path', type=str, default='./weights/sc/', help='Path to save the model weights (default: ./weights/)')
args = parser.parse_args()

# Dataset selection
if args.dataset == 'speechcommands':
    # We'll create a MelSpectrogram transform
    sample_rate = 16000
    audio_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=64,            # you can experiment with different #mel-bins
        n_fft=1024,
        hop_length=256
    )

    # Create training / validation / testing datasets
    #train_dataset = SubsetSC(root="./data_speechcommands", subset="training", transform=audio_transform)
    #val_dataset   = SubsetSC(root="./data_speechcommands", subset="validation", transform=audio_transform)
    test_dataset  = SubsetSC(root="./data_speechcommands", subset="testing", transform=audio_transform)

    # There are 35 labels officially, but only some samples are "commands".
    # If you want to limit to e.g. 10 classes, you must filter them or define
    # a custom label set. We'll assume 35 for simplicity:
    num_classes = 35
    input_channels = 1



# -------------------------------
# Create DataLoaders
# -------------------------------
#train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
# Create a generator and set a fixed seed
generator = torch.Generator()
generator.manual_seed(42)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)


print(' --- Data loaded ---', flush=True)
# Load the model weights
# Initialize model, optimizer, and AMC-Loss

class_names_cifar10 = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

class_names_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class_names_sc = [
        'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four',
        'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
        'tree', 'two', 'up', 'wow', 'yes', 'zero', 'backward', 'forward',
        'follow', 'learn', 'visual'
    ]

if args.dataset == 'cifar10':
    input_channels = 3
    class_names = class_names_cifar10
    num_classes = 10
elif args.dataset == 'mnist':
    input_channels = 1
    class_names = class_names_mnist
    num_classes = 10
elif args.dataset == 'speechcommands':
    input_channels = 1
    class_names = class_names_sc
    num_classes = 35

path = './weights/' + 'sc_re/paper_model_without_amc_ep50_re4.pth'
model_paper = PaperModel(input_channels=input_channels, num_classes=num_classes)  # Reinitialize the model
#model_paper = get_model('resnet18', input_channels, 10, False)
#model_paper = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
model_paper.load_state_dict(torch.load(path, map_location=device))
model_paper = model_paper.to(device)
model_paper.eval()  # Set the model to evaluation mode

path = './weights/' + 'sc_re/paper_model_with_amcpair_ang_0.5_lambda_0.2_re4.pth'
model_paper_amc = PaperModel(input_channels=input_channels, num_classes=num_classes) 
#model_paper_amc = get_model('resnet18', input_channels, 10, False)  # Reinitialize the model
#model_paper_amc = define_tsnet(name=args.model_type, num_class=num_classes, cuda=(device=='cuda'))
model_paper_amc.load_state_dict(torch.load(path, map_location=device))
model_paper_amc = model_paper_amc.to(device)
model_paper_amc.eval()  # Set the model to evaluation mode
print("Model weights loaded and ready for inference.")



# Evaluate on Test Set
evaluate_model(model_paper, test_loader, device)
evaluate_model(model_paper_amc, test_loader, device)
# #print('--- run visualization ---')
visualize_latent_space(model_paper, test_loader, device, class_names, args.dataset)
visualize_latent_space(model_paper_amc, test_loader, device, class_names, args.dataset, use_amc_loss=True)
idx = 32
batch = next(iter(test_loader))
#visualize_grad_cam_sc(model_paper, batch, device, class_names, idx, args.dataset)
#visualize_grad_cam_sc(model_paper_amc, batch, device, class_names, idx, args.dataset, use_amc_loss=True)

#print('--- run Infidelity ---')
# Compute infidelity for the test dataset
infidelity_score = compute_infidelity(model_paper, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score: ', infidelity_score)
infidelity_score = compute_infidelity(model_paper_amc, test_loader, device, noise_std=0.003, perturbations=50)
#print('Infidelity score (AMC): ', infidelity_score)
#infidelity_score = compute_infidelity_sample(model_paper, test_loader, device, idx, noise_std=0.003, perturbations=50)
#infidelity_score = compute_infidelity_sample(model_paper_amc, test_loader, device, idx, noise_std=0.003, perturbations=50)

