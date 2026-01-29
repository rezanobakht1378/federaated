"""pytorchexample: A Flower / PyTorch app using torchvision MNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class Net(nn.Module):
    """Model for MNIST."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition MNIST data using torchvision."""
    
    # Transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full MNIST dataset
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split into partitions (IID)
    partition_size = len(full_dataset) // num_partitions
    indices = list(range(partition_id * partition_size, 
                        (partition_id + 1) * partition_size if partition_id < num_partitions - 1 
                        else len(full_dataset)))
    
    # Create subset for this partition
    partition = torch.utils.data.Subset(full_dataset, indices)
    
    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(partition))
    test_size = len(partition) - train_size
    train_dataset, test_dataset = random_split(partition, [train_size, test_size])
    
    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset():
    """Load full MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    return DataLoader(test_dataset, batch_size=128, shuffle=False)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            else:
                images = batch[0].to(device)
                labels = batch[1].to(device)
            
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            # Check if batch is a dictionary or tuple/list
            if isinstance(batch, dict):
                # If it's a dictionary (from HuggingFace datasets)
                images = batch["image"].to(device)  # Note: MNIST uses "image" not "images"
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # If it's a tuple/list (from torchvision: (images, labels))
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            else:
                # Try to handle generically
                try:
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                except (IndexError, AttributeError) as e:
                    raise ValueError(f"Unable to unpack batch. Batch type: {type(batch)}, contents: {batch}") from e
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
