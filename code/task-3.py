import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_val_loaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Get the training and validation data loaders for the fruits dataset

    Returns:
        A tuple containing the training and validation data loaders, as well as the class names
    """
    # Transform dataset to make compatible with the models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            # Mean and standard deviation of the ImageNet dataset
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    foldername = 'task-3-fruits'
    dataset = datasets.ImageFolder(foldername, transform=transform)

    # Split dataset
    split = 0.7
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    class_names = dataset.classes

    return train_loader, val_loader, class_names


def get_alexnet(num_classes: int) -> models.AlexNet:
    """
    Get the AlexNet model, with the final classifier layer adjusted

    Args:
        num_classes: The new number of classes in the final layer

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
    alexnet.classifier[-1] = torch.nn.Linear(num_features, num_classes)

    alexnet = alexnet.to(DEVICE)
    return alexnet


def get_googlenet(num_classes: int) -> models.GoogLeNet:
    """
    Get the GoogLeNet model, with the final classifier layer adjusted

    Args:
        num_classes: The new number of classes in the final layer

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
    googlenet.fc = torch.nn.Linear(num_features, num_classes)

    googlenet = googlenet.to(DEVICE)
    return googlenet


def get_optimiser(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Get the optimiser to be used with the model

    Args:
        model: The neural network model

    Returns:
        The optimiser
    """
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    return optimiser


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


train_loader, val_loader, classes = get_train_val_loaders()
criterion = torch.nn.CrossEntropyLoss()
googlenet = get_googlenet(len(classes))
train(googlenet, train_loader, criterion, get_optimiser(googlenet))