import argparse
from time import time
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from loss import CrossEntropyWithGradientPenalty
from model import LeNet
from train import train, validate

torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--criterion', '-c', type=str, default='ce')

args = parser.parse_args()

m = LeNet().to('cuda' if torch.cuda.is_available() else 'cpu')

alpha_schedule = [0, 0, 0, 0, 0.025, 0.05, 0.075, 0.1, 0.1, 0.1]
if args.criterion == 'gp':
    criterion = CrossEntropyWithGradientPenalty(m, alpha_schedule=alpha_schedule, bad_concept_labels=torch.tensor([0, 1]))
elif args.criterion == 'ce':
    criterion = CrossEntropyLoss()
else:
    raise ValueError(f'Invalid criterion: {args.criterion}. Options are "ce" and "gp".')
optimizer = optim.Adam(m.parameters(), lr=1e-4)
batch_size = 32
epochs = len(alpha_schedule)

print('criterion:', args.criterion)
if not isinstance(criterion, CrossEntropyLoss):
    print('alpha schedule:', alpha_schedule)
print('optimizer:', optimizer, 'lr:', optimizer.param_groups[0]['lr'])
print('batch_size:', batch_size)
print('epochs:', epochs)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1308,), (0.3016,))
])

full_train_dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=mnist_transform
)
shuffled_train_indices = torch.randperm(len(full_train_dataset))
train_dataset = Subset(full_train_dataset, shuffled_train_indices[:50000])
ft_dataset = Subset(full_train_dataset, shuffled_train_indices[50000:])

full_val_dataset = MNIST(
    root='./data',
    train=False,
    download=True,
    transform=mnist_transform
)
shuffled_val_indices = torch.randperm(len(full_val_dataset))
val_dataset = Subset(full_val_dataset, shuffled_val_indices[:len(full_val_dataset) // 2])
test_dataset = Subset(full_val_dataset, shuffled_val_indices[len(full_val_dataset) // 2:])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)
ft_loader = DataLoader(
    ft_dataset,
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

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

best_err1 = 100
best_err5 = 100

start = time()

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

print('\ncriterion:', args.criterion)
if not isinstance(criterion, CrossEntropyLoss):
    print('alpha schedule:', alpha_schedule)
print('optimizer:', optimizer, 'lr:', optimizer.param_groups[0]['lr'])
print('batch_size:', batch_size)
print('epochs:', epochs)
print(f'total time: {time() - start:.3f} seconds')

name = f'{args.criterion}_{"_".join([str(x) for x in alpha_schedule])}'
torch.save(m.state_dict(), f'{name}.pth')

"""FINE-TUNING"""
optimizer = optim.Adam(m.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()
best_err1 = 100
best_err5 = 100

start = time()

# epoch loop
for epoch in range(0, epochs):

    # train for one epoch
    train_loss = train(
        ft_loader,
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

    print('FT Current best error rate (top-1 and top-5 error):', best_err1, best_err5, '\n')
print('FT Best error rate (top-1 and top-5 error):', best_err1, best_err5)

print('\nFT criterion: CrossEntropyLoss')
print('optimizer:', optimizer, 'lr:', optimizer.param_groups[0]['lr'])
print('batch_size:', batch_size)
print('epochs:', epochs)
print(f'total time: {time() - start:.3f} seconds')

name = f'{args.criterion}_{"_".join([str(x) for x in alpha_schedule])}'
torch.save(m.state_dict(), f'ft_{name}.pth')