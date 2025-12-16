import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# Transform data
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

split = 0.7
train_size = int(split * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

class_names = dataset.classes
print(class_names)