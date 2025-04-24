# `psmi`

# Copyright 2024-present Laboratoire d'Informatique de Polytechnique.
# License LGPL-3.0

import os

import torch

data_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data"))
hidden_reps_path = os.path.join(data_root, "hidden_reps.pt")
labels_path = os.path.join(data_root, "labels.pt")


def compute():
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # 1. Download a pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # Modify model to access hidden representation
    model.fc = torch.nn.Identity()  # Remove final classification layer

    # 2. Download only CIFAR-10 test set
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    cifar10_test = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    dataloader = DataLoader(cifar10_test, batch_size=32, shuffle=False)

    # 3. Evaluate model on CIFAR-10 test set and get hidden representations with progress bar
    hidden_reps = []
    labels = []
    with torch.no_grad():
        for images, lab in tqdm(dataloader, desc="Evaluating"):
            hidden_rep = model(images)
            hidden_reps.append(hidden_rep)
            labels.append(lab)

    # Flatten hidden representations into a single tensor
    hidden_reps = torch.cat(hidden_reps)
    labels = torch.cat(labels)
    torch.save(hidden_reps, hidden_reps_path)
    torch.save(labels, labels_path)


def get_test_data() -> torch.Tensor:

    if not os.path.isfile(hidden_reps_path) or not os.path.isfile(labels_path):
        compute()

    return torch.load(hidden_reps_path, weights_only=True), torch.load(
        labels_path, weights_only=True
    )
