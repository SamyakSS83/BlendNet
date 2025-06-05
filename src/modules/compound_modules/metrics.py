import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from typing import Union

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)


class PearsonR(nn.Module):
    """
    Calculates Pearson's correlation coefficient
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        shifted_x = preds - torch.mean(preds, dim=0)
        shifted_y = targets - torch.mean(targets, dim=0)
        sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
        sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

        pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
        pearson = torch.clamp(pearson, min=-1, max=1)
        pearson = pearson.mean()
        return pearson

def cov_loss(x):
    """
    Computes the covariance loss, which encourages diverse representations
    
    Args:
        x: tensor of shape [batch_size, feature_dim]
        
    Returns:
        loss: scalar tensor representing the covariance loss
    """
    batch_size, feature_dim = x.shape
    
    # Center the features
    x = x - torch.mean(x, dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.matmul(x.t(), x) / (batch_size - 1)
    
    # Compute loss (minimize off-diagonal elements)
    off_diag = torch.triu(cov, diagonal=1)
    loss = torch.sum(off_diag ** 2)
    
    return loss

def uniformity_loss(x, t=2):
    """
    Computes the uniformity loss, which encourages uniform distribution of features
    
    Args:
        x: tensor of shape [batch_size, feature_dim]
        t: temperature parameter
        
    Returns:
        loss: scalar tensor representing the uniformity loss
    """
    batch_size = x.shape[0]
    
    # Compute pairwise distances
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x_normalized = x / (x_norm + 1e-8)
    
    # Compute exp(-t * ||u-v||^2)
    pairwise_distance = torch.matmul(x_normalized, x_normalized.t())
    pairwise_distance = 2 - 2 * pairwise_distance  # Euclidean distance squared: ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u·v = 2 - 2*u·v (when ||u||=||v||=1)
    
    # Apply negative exponent and mask out diagonals
    mask = torch.eye(batch_size, device=x.device).bool()
    loss = torch.exp(-t * pairwise_distance[~mask]).mean().log()
    
    return loss

class RMSELoss(nn.Module):
    """
    Root Mean Square Error Loss
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, preds, targets):
        return torch.sqrt(self.mse(preds, targets))

class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        return torch.mean(torch.abs(preds - targets))
        
class R2Score(nn.Module):
    """
    R² coefficient of determination score
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        target_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
