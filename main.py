import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from loss import CrossEntropyWithGradientPenalty
from model import LeNet
from train import train, validate

m = LeNet().to('cuda' if torch.cuda.is_available() else 'cpu')

# criterion = CrossEntropyWithGradientPenalty(m, alpha=0.0)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(m.parameters(), lr=1e-4)
batch_size = 32
epochs = 10

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1308,), (0.3016,))
])

train_dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=mnist_transform
)

val_dataset = MNIST(
    root='./data',
    train=False,
    download=True,
    transform=mnist_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

best_err1 = 100
best_err5 = 100

# epoch loop
for epoch in range(0, epochs):

    # train for one epoch
    train_loss = train(
        train_loader,
        m,
        criterion,
        optimizer,
        epoch,
        epochs
    )

    # evaluate on validation set
    err1, err5, val_loss = validate(
        val_loader,
        m,
        criterion,
        epoch,
        epochs
    )

    # remember best prec@1 and save checkpoint
    is_best = err1 <= best_err1
    best_err1 = min(err1, best_err1)
    if is_best:
        best_err5 = err5

    print('Current best error rate (top-1 and top-5 error):', best_err1, best_err5, '\n')
print('Best error rate (top-1 and top-5 error):', best_err1, best_err5)