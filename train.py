from time import time
import torch
from sklearn.metrics import classification_report

from loss import CrossEntropyWithGradientPenalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):

    """Computes and stores an average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def error_rate(output, target, topk=(1,)):

    """Computes the top-k error rate for the specified values of k."""

    maxk = max(topk) # largest k we'll need to work with
    batch_size = target.size(0) # determine batch size

    # get maxk best predictions for each item in the batch, both values and indices
    _, pred = output.topk(maxk, 1, True, True)

    # reshape predictions and targets and compare them element-wise
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk: # for each top-k accuracy we want

        # num correct
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # num incorrect
        wrong_k = batch_size - correct_k
        # as a percentage
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

# training function - 1 epoch
def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    print_freq = 100,
    verbose = True
):

    # track average and worst losses
    losses = AverageMeter()
    times = []

    # set training mode
    model.train()
    
    # step the criterion scheduler
    if isinstance(criterion, CrossEntropyWithGradientPenalty):
        criterion.step()

    # iterate over data - automatically shuffled
    for i, (images, labels) in enumerate(train_loader):
        
        start = time()

        # put batch of image tensors on GPU
        images = images.to(device)
        # put batch of label tensors on GPU
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # model output
        outputs = model(images)

        # loss computation
        loss = criterion(outputs, labels)

        # back propagation
        loss.backward()

        # update model parameters
        optimizer.step()

        # update meter with the value of the loss once for each item in the batch
        losses.update(loss.item(), images.size(0))

        # logging during epoch
        if i % print_freq == 0 and verbose == True:
            print(
                f'Epoch: [{epoch+1}/{epochs}][{i:4}/{len(train_loader)}]\t'
                f'Loss: {losses.val:.4f} ({losses.avg:.4f} on avg)'
            )
            
        times.append(time() - start)

    # log again at end of epoch
    print(f'\n* Epoch: [{epoch+1}/{epochs}]\tTrain loss: {losses.avg:.3f}\n')
    print(f'Criterion: {criterion}, average time: {sum(times)/len(times):.3f} seconds\n')

    return losses.avg

# val function
def validate(
    val_loader,
    model,
    criterion,
    epoch,
    epochs,
    print_freq = 100,
    verbose = True
):

    # track average and worst losses and batch-wise top-1 and top-5 accuracies
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set evaluation mode
    model.eval()
    
    y_true = []
    y_pred = []

    # iterate over data - automatically shuffled
    for i, (images, labels) in enumerate(val_loader):

        # put batch of image tensors on GPU
        images = images.to(device)
        # put batch of label tensors on GPU
        labels = labels.to(device)

        # model output
        output = model(images)

        # loss computation
        loss = criterion(output, labels)

        # top-1 and top-5 accuracy on this batch
        err1, err5, = error_rate(output.data, labels, topk=(1, 5))

        # update meters with the value of the loss once for each item in the batch
        losses.update(loss.item(), images.size(0))
        # update meters with top-1 and top-5 accuracy on this batch once for each item in the batch
        top1.update(err1.item(), images.size(0))
        top5.update(err5.item(), images.size(0))
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

        # logging during epoch
        if i % print_freq == 0 and verbose == True:
            print(
                f'Test (on val set): [{epoch+1}/{epochs}][{i:4}/{len(val_loader)}]\t'
                f'Loss: {losses.val:.4f} ({losses.avg:.4f} on avg)\t'
                f'Top-1 err: {top1.val:.4f} ({top1.avg:.4f} on avg)\t'
                f'Top-5 err: {top5.val:.4f} ({top5.avg:.4f} on avg)'
            )

    # logging for end of epoch
    print(
        f'\n* Epoch: [{epoch+1}/{epochs}]\t'
        f'Test loss: {losses.avg:.3f}\t'
        f'Top-1 err: {top1.avg:.3f}\t'
        f'Top-5 err: {top5.avg:.3f}\n'
    )
    
    print(classification_report(y_true, y_pred))

    # avergae top-1 and top-5 accuracies batch-wise, and average loss batch-wise
    return top1.avg, top5.avg, losses.avg