import torch
import torch.nn as nn
from model import LeNet

class CrossEntropyWithGradientPenalty(nn.Module):
    def __init__(self, model, alpha_schedule):
        """
        Initializes the custom loss function.
        
        Parameters:
        - model: The neural network model.
        - alpha: Weight for the gradient penalty term. Higher values increase the emphasis on minimizing the gradient.
        """
        super(CrossEntropyWithGradientPenalty, self).__init__()
        self.model = model
        self.alpha_schedule = alpha_schedule
        self.alpha_index = 0
        self.alpha = self.alpha_schedule[self.alpha_index]
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Computes the combined loss (cross entropy + gradient penalty).
        
        Parameters:
        - outputs: Model outputs (logits).
        - targets: Ground-truth labels.
        
        Returns:
        - loss: The combined loss value.
        """
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(outputs, targets)
        
        # Compute gradients of the loss with respect to the model parameters
        grads = torch.autograd.grad(
            ce_loss, self.model.parameters(), create_graph=True
        )
        
        # Compute the norm of the gradients
        grad_norm = sum(
            torch.norm(g)**2 for g in grads if g is not None
        )
        
        # Combined loss: cross-entropy loss + alpha * gradient penalty
        loss = ce_loss + self.alpha * grad_norm
        
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