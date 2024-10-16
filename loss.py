import torch
import torch.nn as nn

class CrossEntropyWithGradientPenalty(nn.Module):
    def __init__(self, model, alpha=0.1):
        """
        Initializes the custom loss function.
        
        Parameters:
        - model: The neural network model.
        - alpha: Weight for the gradient penalty term. Higher values increase the emphasis on minimizing the gradient.
        """
        super(CrossEntropyWithGradientPenalty, self).__init__()
        self.model = model
        self.alpha = alpha
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
            torch.norm(g.detach())**2 for g in grads if g is not None
        )
        
        # Combined loss: cross-entropy loss + alpha * gradient penalty
        loss = ce_loss + self.alpha * grad_norm
        
        return loss
