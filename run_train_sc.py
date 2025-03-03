import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchaudio
import torchaudio.transforms as T
import os
import ssl
import argparse
from model import PaperModel
from train import train_model, AMCLossWithPairing, collate_fn
from evaluate import evaluate_model

# -----------------------------------
# 1. Subclassing the SpeechCommands
#    dataset to split into train/val/test
# -----------------------------------
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



# -----------------------------------
# 3. Main Script (with arguments)
# -----------------------------------
if __name__ == "__main__":
    # Disable SSL verification for dataset downloading
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser(description="Train a model with optional AMC-Loss on various datasets.")
    parser.add_argument('--dataset', type=str, default='speechcommands',
                        choices=['speechcommands'],
                        help='Dataset to use (default: speechcommands)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training without AMC-Loss')
    parser.add_argument('--num_epochs_amc', type=int, default=100, help='Number of epochs for training with AMC-Loss')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--angular', type=float, default=1, help='AMC-loss angular (default: 1)')
    parser.add_argument('--lambda_', type=float, default=0.1,help='AMC-loss angular (default: 0.1)')
    parser.add_argument('--save_path', type=str, default='./weights/sc', help='Path to save the model weights')
    parser.add_argument('--note', type=str, default='sgd', help='Note to save the model weights (default: ./weights/cifar10/)')
    args = parser.parse_args()

    # Decide the device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Create transforms / Datasets
    # -------------------------------
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
        train_dataset = SubsetSC(root="./data_speechcommands", subset="training", transform=audio_transform)
        val_dataset   = SubsetSC(root="./data_speechcommands", subset="validation", transform=audio_transform)
        test_dataset  = SubsetSC(root="./data_speechcommands", subset="testing", transform=audio_transform)

        # There are 35 labels officially, but only some samples are "commands".
        # If you want to limit to e.g. 10 classes, you must filter them or define
        # a custom label set. We'll assume 35 for simplicity:
        num_classes = 35
        input_channels = 1

    

    # -------------------------------
    # Create DataLoaders
    # -------------------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optionally create a val_loader if you want to do validation checks
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    print('--- Data loaded ---')

    # -------------------------------
    # Initialize Model and Optimizer
    # -------------------------------
    model_paper = PaperModel(input_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model_paper.parameters(), lr=args.lr)

    # # -------------------------------
    # # Train WITHOUT AMC-Loss
    # # -------------------------------
    print("Training without AMC-Loss")
    path = os.path.join(args.save_path, f"paper_model_without_amc")
    train_model(model_paper, 
                path=path, 
                criterion=F.cross_entropy, 
                optimizer=optimizer, 
                dataloader=train_loader,
                dataloader_val=val_loader,
                num_epochs=args.num_epochs,
                use_amc_loss=False
                ) 

    # Save the model
    final_path = os.path.join(args.save_path, f"paper_model_without_amc_ep{args.num_epochs}_{args.note}.pth")
    torch.save(model_paper.state_dict(), final_path)
    print(f"Model weights saved at {final_path}")

    # Evaluate on Test Set
    evaluate_model(model_paper, test_loader, device)

    # -------------------------------
    # Train WITH AMC-Loss
    # -------------------------------
    print("Training with AMC-Loss")
    angular_margin = args.angular
    lambda_ = args.lambda_
    criterion_amc = AMCLossWithPairing(angular_margin=angular_margin, lambda_=lambda_)

    model_paper_amc = PaperModel(input_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model_paper_amc.parameters(), lr=args.lr)

    path = os.path.join(
        args.save_path, 
        f"paper_model_with_amcpair_ang_{angular_margin}_lambda_{lambda_}"
    )

    train_model(model_paper_amc, 
                path=path, 
                criterion=criterion_amc, 
                optimizer=optimizer, 
                dataloader=train_loader, 
                dataloader_val=val_loader,
                num_epochs=args.num_epochs_amc,
                use_amc_loss=True,
                ) 

    # Save the model
    final_path = os.path.join(
        args.save_path, 
        f"paper_model_with_amcpair_ang_{angular_margin}_lambda_{lambda_}_{args.note}.pth"
    )
    torch.save(model_paper_amc.state_dict(), final_path)
    print(f"Model weights saved at {final_path}")

    # Evaluate on Test Set
    evaluate_model(model_paper_amc, test_loader, device)
