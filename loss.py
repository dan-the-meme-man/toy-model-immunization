import torch
import torch.nn as nn
from model import LeNet

class CrossEntropyWithGradientPenalty(nn.Module):
    def __init__(self, model, alpha_schedule, bad_concept_labels = None):
        """
        Initializes the custom loss function.
        
        Parameters:
        - model: The neural network model.
        - alpha: Weight for the gradient penalty term. Higher values increase the emphasis on minimizing the gradient.
        """
        super(CrossEntropyWithGradientPenalty, self).__init__()
        self.model = model
        self.alpha_schedule = alpha_schedule
        self.alpha_index = -1
        if bad_concept_labels is not None:
            self.bad_concept_labels = bad_concept_labels
        else:
            self.bad_concept_labels = torch.tensor([])
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def good_bad_split(self, outputs, targets):
        
        bad_indices = torch.isin(targets, self.bad_concept_labels)
        good_indices = torch.logical_not(bad_indices)
        
        bad_targets = targets[bad_indices]
        good_targets = targets[good_indices]
        
        bad_outputs = outputs[bad_indices]
        good_outputs = outputs[good_indices]
        
        return good_outputs, good_targets, bad_outputs, bad_targets

    def forward(self, outputs, targets):
        """
        Computes the combined loss (cross entropy + gradient penalty).
        
        Parameters:
        - outputs: Model outputs (logits).
        - targets: Ground-truth labels.
        
        Returns:
        - loss: The combined loss value.
        """
        
        good_outputs, good_targets, bad_outputs, bad_targets = self.good_bad_split(outputs, targets)
        
        # standard cross-entropy loss
        good_ce_loss = self.cross_entropy(good_outputs, good_targets)
        bad_ce_loss = self.cross_entropy(bad_outputs, bad_targets)
        
        # get gradients just for bad outputs
        bad_grads = torch.autograd.grad(
            bad_ce_loss, self.model.parameters(), create_graph=True
        )
        
        # Compute the norm of the bad gradients
        bad_grad_norm = sum(
            torch.norm(g)**2 for g in bad_grads if g is not None
        )
        
        # Combined loss: cross entropy on good outputs - cross entropy on bad outputs + alpha * gradient norm
        loss = good_ce_loss - bad_ce_loss + self.alpha * bad_grad_norm
        
        return loss
    
    def step(self):
        self.alpha_index += 1
        self.alpha = self.alpha_schedule[self.alpha_index]

def main():
    inputs = torch.randn(32, 1, 28, 28).to('cuda')
    targets = torch.randint(0, 10, (32,)).to('cuda')
    
    m = LeNet().to('cuda')
    
    outputs = m(inputs)
    
    criterion = CrossEntropyWithGradientPenalty(m, alpha=0.1)
    
    loss = criterion(outputs, targets)
    
    loss.backward()
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    
    optimizer.step()
    
if __name__ == '__main__':
    main()