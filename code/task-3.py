
# NOTE: The 'utils' module imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'utils' folder within this directory
# The 'torch' and 'torchvision' modules were NOT created by our team -- they are PyTorch libraries

import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import confusion_matrix, plot_loss, plot_acc, plot_confusion_matrix, plot_lr_val_accuracies, plot_lr_val_losses


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _valid_ratio(ratio: tuple[float, ...]) -> bool:
    """
    Check if the ratio in the tuple is valid

    Args:
        ratio: Ratio that should add to 1, with only non-negative values

    Returns:
        A True or False value depending on if the ratio is valid
    """
    return all(x >= 0 for x in ratio) and sum(ratio) == 1.0


def get_train_val_test_loaders(foldername: str, ratio: tuple[float, float, float], batch_size: int = 4, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Get the training, validation, and test data loaders for the fruits dataset

    Args:
        seed: Random seed for reproducibility
        foldername: Name of the dataset folder 
        ratio: Desired split -- e.g. (0.7, 0.3, 0.0) means a 70% train / 30% validation / 0% test split
        batch_size: The batch size for the data loaders

    Returns:
        A tuple containing the training, validation, and test data loaders, as well as the class names
    """
    if not _valid_ratio(ratio):
        raise ValueError('Invalid ratio!')

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
    test_size = int(ratio[2] * len(dataset))
    val_size = int(ratio[1] * len(dataset))
    train_size = len(dataset) - (val_size + test_size)
    sizes = [train_size, val_size, test_size]
    names = ['train', 'val', 'test']
    
    # Get non-zero sizes
    non_zero_sizes = []
    non_zero_names = []
    for name, size in zip(names, sizes):
        if size > 0:
            non_zero_sizes.append(size)
            non_zero_names.append(name)

    splits = random_split(dataset, non_zero_sizes, generator=torch.Generator().manual_seed(seed))

    # Create data loaders
    loaders = {}
    for i, name in enumerate(non_zero_names):
        shuffle = (name == 'train')  # only shuffle training set
        loaders[name] = DataLoader(splits[i], batch_size=batch_size, shuffle=shuffle)

    class_names = dataset.classes

    return loaders.get('train'), loaders.get('val'), loaders.get('test'), class_names


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


def get_optimiser(model: torch.nn.Module, name: str, **kwargs) -> torch.optim.Optimizer:
    """
    Get an optimiser for the model

    Args:
        model: The neural network model
        name: Name of the optimisr
        kwargs: Parameters of the optimiser

    Returns:
        A new optimiser
    """
    if name == 'SGD':
        Optimiser = torch.optim.SGD
    elif name == 'Adam':
        Optimiser = torch.optim.Adam
    else:
        raise ValueError(f'Unknown optimiser: {name}')
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    return Optimiser(params, **kwargs)


def get_criterion(name: str, **kwargs) -> torch.nn.Module:
    """
    Get a criterion / loss function

    Args:
        name: Name of the criterion
        kwargs: Parameters of the criterion

    Returns:
        A new criterion
    """
    if name == 'CrossEntropyLoss':
        Criterion = torch.nn.CrossEntropyLoss
    elif name == 'MSELoss':
        Criterion = torch.nn.MSELoss
    else:
        raise ValueError(f'Unknown criterion: {name}')
    
    return Criterion(**kwargs)


def _train(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimiser: torch.optim.Optimizer) -> tuple[float, float]:
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
def _validate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module) -> tuple[float, float]:
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


def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: torch.nn.Module, optimiser: torch.optim.Optimizer, epochs: int) -> dict[str, list[float]]:
    """
    Train the model on the training data and validate

    Args:
        model: The neural network model
        train_loader: The training data loader
        val_loader: The validation data loader
        criterion: The loss function to be used
        optimiser: The optimiser to be used
        epochs: The number of iterations to train for

    Returns:
        The losses and accuracy of the model during the different epochs on the training and validation data
    """
    print(f'=== Training {type(model).__name__} ===')

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(1, epochs+1):
        train_loss, train_acc = _train(model, train_loader, criterion, optimiser)
        val_loss, val_acc = _validate(model, val_loader, criterion)

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

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, criterion: torch.nn.Module) -> dict[str, float | list[float]]:
    """
    Evaluate the model on the test data

    Args:
        model: The neural nework model
        test_loader: The test data loader
        criterion: The loss function to be used

    Returns:
        The test loss and accuracy, and all of the predictions and actual labels
    """
    print(f'=== Evaluating {type(model).__name__} ===')

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)

        all_labels += labels.tolist()
        all_preds += preds.tolist()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    test_loss, test_acc = running_loss / len(test_loader), correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n')

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'all_labels': all_labels,
        'all_preds': all_preds
    }


if __name__ == "__main__":
    # Train AlexNet and GoogleLeNet on the fruits dataset
    foldername = 'task-3-fruits'
    split = (0.7, 0.3, 0)  # (train, validation, test)
    train_loader, val_loader, _, classes = get_train_val_test_loaders(foldername, ratio=split)
    num_classes = len(classes)

    print(f'Dataset: "{foldername}" | Classes: {classes}\n')

    iterations = 30

    criterion = {
       'name': 'CrossEntropyLoss'
    }
    optimiser = {
        'name': 'SGD',
        'lr': 1e-3,
    }

    alexnet = get_alexnet(output_classes=num_classes)
    alexnet_res = train_model(
        model=alexnet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=get_criterion(**criterion),
        optimiser=get_optimiser(alexnet, **optimiser),
        epochs=iterations
    )
    alexnet_eval = evaluate_model(alexnet, val_loader, get_criterion(**criterion))

    plot_loss(alexnet_res['train_losses'], alexnet_res['val_losses'])
    plot_acc(alexnet_res['train_accs'], alexnet_res['val_accs'])
    plot_confusion_matrix(confusion_matrix(alexnet_eval['all_labels'], alexnet_eval['all_preds'], num_classes), classes)

    googlenet = get_googlenet(output_classes=num_classes)
    googlenet_res = train_model(
        model=googlenet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=get_criterion(**criterion),
        optimiser=get_optimiser(googlenet, **optimiser),
        epochs=iterations
    )
    googlenet_eval = evaluate_model(googlenet, val_loader, get_criterion(**criterion))
    
    plot_loss(googlenet_res['train_losses'], googlenet_res['val_losses'])
    plot_acc(googlenet_res['train_accs'], googlenet_res['val_accs'])
    plot_confusion_matrix(confusion_matrix(googlenet_eval['all_labels'], googlenet_eval['all_preds'], num_classes), classes)
   
    # Learning rate comparison (AlexNet + GoogLeNet)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    alexnet_hist = {}
    for lr in learning_rates:
        print(f'========== AlexNet | Learning rate: {lr} ==========\n')
        model_lr = get_alexnet(output_classes=num_classes)

        res = train_model(
            model=model_lr,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=get_criterion(**criterion),
            optimiser=get_optimiser(model_lr, name=optimiser['name'], lr=lr),
            epochs=iterations
        )
        alexnet_hist[lr] = res

    plot_lr_val_losses(alexnet_hist, title="AlexNet: Validation Loss vs Epoch")
    plot_lr_val_accuracies(alexnet_hist, title="AlexNet: Validation Accuracy vs Epoch")

    googlenet_hist = {}
    for lr in learning_rates:
        print(f'========== GoogLeNet | Learning rate: {lr} ==========\n')
        model_lr = get_googlenet(output_classes=num_classes)

        res = train_model(
            model=model_lr,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=get_criterion(**criterion),
            optimiser=get_optimiser(model_lr, name=optimiser['name'], lr=lr),
            epochs=iterations
        )
        googlenet_hist[lr] = res

    plot_lr_val_losses(googlenet_hist, title="GoogLeNet: Validation Loss vs Epoch")
    plot_lr_val_accuracies(googlenet_hist, title="GoogLeNet: Validation Accuracy vs Epoch")
      
