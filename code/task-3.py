
# NOTE: The 'utils' module imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'utils' folder within this directory
# The 'torch' and 'torchvision' modules were NOT created by our team -- they are PyTorch libraries

import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import plot_loss, plot_acc

# TODO
# Maybe improve image dataset -- remove images that do not look similar with the rest of the images
# Improve results visualisation -- check Transfer Learning example in week 5 lecture notes on Moodle


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_val_loaders(foldername: str, ratio: float, seed: int = 42) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Get the training and validation data loaders for the fruits dataset

    Args:
        seed: Random seed for reproducibility
        foldername: Name of the dataset folder 
        ratio: Desired split -- e.g. 0.7 means a 70% / 30% split

    Returns:
        A tuple containing the training and validation data loaders, as well as the class names
    """
    # Transform dataset to make compatible with the models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # The original AlexNet paper points to using 227, but the PyTorch implementation corrects this to 224
        transforms.ToTensor(),
        transforms.Normalize(
            # Mean and standard deviation of the ImageNet dataset
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(foldername, transform=transform)

    # Split dataset
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    class_names = dataset.classes

    return train_loader, val_loader, class_names


def get_alexnet(output_classes: int) -> models.AlexNet:
    """
    Get the AlexNet model, with the final classifier layer adjusted

    Args:
        output_classes: The new number of classes in the final layer

    Returns:
        The AlexNet model
    """
    # Setup AlexNet
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Freeze feature extractors
    for param in alexnet.features.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    num_features = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = torch.nn.Linear(num_features, output_classes)

    alexnet = alexnet.to(DEVICE)
    return alexnet


def get_googlenet(output_classes: int) -> models.GoogLeNet:
    """
    Get the GoogLeNet model, with the final classifier layer adjusted

    Args:
        output_classes: The new number of classes in the final layer

    Returns:
        The GoogLeNet model
    """
    # Setup GoogLeNet
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    # Freeze layers
    for param in googlenet.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    num_features = googlenet.fc.in_features
    googlenet.fc = torch.nn.Linear(num_features, output_classes)

    googlenet = googlenet.to(DEVICE)
    return googlenet


def train(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimiser: torch.optim.Optimizer) -> tuple[float, float]:
    """
    Train the model for a single epoch

    Args:
        model: The neural network model
        loader: The training data loader
        criterion: The loss function to be used
        optimiser: The optimiser to be used

    Returns:
        The training loss and accuracy of the model during the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module) -> tuple[float, float]:
    """
    Validate the model for a single epoch

    Args:
        model: The neural network model
        loader: The validation data loader
        criterion: The loss function to be used

    Returns:
        The validation loss and accuracy of the model during the epoch
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train the model on the training data and validate

    Args:
        model: The neural network model
        train_loader: The training data loader
        val_loader: The validation data loader
        epochs: The number of iterations to train for

    Returns:
        The losses and accuracy of the model during the different epochs on the training and validation data
    """
    print(f'=== Training {type(model).__name__} ===')

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimiser)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f'Epoch {epoch}/{epochs} | '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        )
    
    print()

    return train_losses, train_accs, val_losses, val_accs


if __name__ == "__main__":
    # Train AlexNet and GoogleLeNet on the fruits dataset
    foldername = 'task-3-fruits'
    split = 0.7
    train_loader, val_loader, classes = get_train_val_loaders(foldername, ratio=split)
    num_classes = len(classes)

    print(f'Dataset: "{foldername}" | Classes: {classes}\n')

    iterations = 30

    alexnet = get_alexnet(output_classes=num_classes)
    a_train_losses, a_train_accs, a_val_losses, a_val_accs = train_model(alexnet, train_loader, val_loader, epochs=iterations)

    googlenet = get_googlenet(output_classes=num_classes)
    g_train_losses, g_train_accs, g_val_losses, g_val_accs = train_model(googlenet, train_loader, val_loader, epochs=iterations)
    
    plot_loss(a_train_losses, a_val_losses)
    plot_loss(g_train_losses, g_val_losses)
    plot_acc(a_train_accs, a_val_accs)
    plot_acc(g_train_accs, g_val_accs)
