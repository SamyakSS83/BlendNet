import math
import torch
import itertools
import numpy as np

from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss

class OGBNanLabelBCEWithLogitsLoss(_Loss):
    def __init__(self) -> None:
        super(OGBNanLabelBCEWithLogitsLoss, self).__init__()
        self.bce_loss = BCEWithLogitsLoss()
    def forward(self, pred, target, **kwargs):
        is_labeled = ~torch.isnan(target)

        loss = self.bce_loss(pred[is_labeled], target[is_labeled])
        return loss

class OGBNanLabelMSELoss(_Loss):
    def __init__(self) -> None:
        super(OGBNanLabelMSELoss, self).__init__()
        self.mse_loss = MSELoss()
    def forward(self, pred, target, **kwargs):
        is_labeled = ~torch.isnan(target)

        loss = self.mse_loss(pred[is_labeled], target[is_labeled])
        return loss

class CriticLoss(_Loss):
    def __init__(self) -> None:
        super(CriticLoss, self).__init__()

    def forward(self, z2, reconstruction, **kwargs):
        batch_size, metric_dim, repeats = reconstruction.size()
        z2_norm = F.normalize(z2, dim=1, p=2)[..., None].repeat(1, 1, repeats)
        reconstruction_norm = F.normalize(reconstruction, dim=1, p=2)
        loss = (((z2_norm - reconstruction_norm) ** 2).sum(dim=1)).mean()
        return loss


class BarlowTwinsLoss(_Loss):
    def __init__(self, scale_loss=1 / 32, lambd=3.9e-3, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        "Loss function from the Barlow twins paper from yann lecun"
        super(BarlowTwinsLoss, self).__init__()
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs):
        # normalize repr. along the batch dimension
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD

        N, D = z1.size()

        # cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device=c.device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambd
        loss = c_diff.sum()

        if self.uniformity_reg > 0:
            loss = loss + self.uniformity_reg * self._calc_uniformity_loss(z1)
            loss = loss + self.uniformity_reg * self._calc_uniformity_loss(z2)
        return loss * self.scale_loss

    def _calc_uniformity_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def _calc_variance_loss(self, x, eps=1e-04):
        std_x = torch.sqrt(x.var(dim=0) + eps)
        loss_std = torch.mean(F.relu(1 - std_x))
        return loss_std

    def _calc_covariance_loss(self, x, eps=1e-04):
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.size(0) - 1)
        loss_cov = off_diagonal(cov_x).pow_(2).sum() / x.size(1)
        return loss_cov

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class NTXent(_Loss):
    def __init__(self, temperature=0.5) -> None:
        """Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper

        Parameters:
            temperature: A float value between 0 and 1 that determines how strong the loss gradients are
        """
        super(NTXent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j, **kwargs):
        """Forward pass of the NTXent loss.

        Parameters:
           z_i: hidden vector of shape [batch_size, dim]
           z_j: hidden vector of shape [batch_size, dim]

        Returns:
           loss
        """
        batch_size = z_i.shape[0]
        # concatenate the vectors and L2-normalize them
        representations = torch.cat([z_i, z_j], dim=0)
        representations = F.normalize(representations, dim=1, p=2)

        similarity_matrix = torch.matmul(representations, representations.transpose(0, 1))
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool, device=similarity_matrix.device)).float()
        negatives = similarity_matrix[mask.bool()].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        # target index is always the positive sample at index 0
        targets = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, targets)

        return loss

class NTXentMultiplePositives(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentMultiplePositives, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, anchors, positives, negatives=None):
        """
        Args:
            anchors: Tensor of shape [batch_size, d]
            positives: List of tensors, each of shape [batch_size, d]
            negatives: Optional tensor of shape [n_neg, d]
        """
        device = anchors.device
        batch_size = anchors.size(0)
        
        # Normalize all vectors
        anchors = F.normalize(anchors, dim=1)
        positives = [F.normalize(p, dim=1) for p in positives]
        if negatives is not None:
            negatives = F.normalize(negatives, dim=1)
        
        # Compute similarity between anchors and positives
        pos_similarity = []
        for pos in positives:
            sim = torch.einsum('bd,bd->b', anchors, pos) / self.temperature
            pos_similarity.append(sim)
        
        pos_similarity = torch.stack(pos_similarity, dim=1)  # [batch_size, n_pos]
        
        # If negatives are provided, compute similarity with negatives
        if negatives is not None:
            neg_similarity = torch.matmul(anchors, negatives.t()) / self.temperature  # [batch_size, n_neg]
            # Combine positive and negative similarities
            logits = torch.cat([pos_similarity, neg_similarity], dim=1)
        else:
            # Create negative samples from other examples in the batch
            # For each anchor, all other examples (and their positives) are negatives
            all_samples = torch.cat([anchors] + positives, dim=0)  # [batch_size * (1 + n_pos), d]
            
            # Compute full similarity matrix
            full_similarity = torch.matmul(anchors, all_samples.t()) / self.temperature  # [batch_size, batch_size * (1 + n_pos)]
            
            # Create a mask to identify the positive samples for each anchor
            mask = torch.zeros_like(full_similarity)
            
            # Mark the anchor itself
            mask[:, :batch_size] = torch.eye(batch_size, device=device)
            
            # Mark the positives
            for i, _ in enumerate(positives):
                pos_start_idx = batch_size * (i + 1)
                mask[:, pos_start_idx:pos_start_idx + batch_size] = torch.eye(batch_size, device=device)
            
            # Apply the mask to exclude positives from the logits
            neg_similarity = full_similarity.masked_fill(mask == 1, float('-inf'))
            
            # Create logits by combining positive similarities with masked negative similarities
            logits = torch.cat([pos_similarity, neg_similarity], dim=1)
        
        # Our targets are always the positives (the first n_pos elements in each row)
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute the loss
        loss = self.criterion(logits, targets)
        
        return loss

def uniformity_loss(x, t=2):
    """
    Uniformity loss from Wang & Isola, "Understanding Contrastive Representation Learning Through Alignment and Uniformity on the Hypersphere"
    https://arxiv.org/abs/2005.10242
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def cov_loss(x):
    """Calculate covariance loss to encourage diverse representations.
    Based on VICReg implementation.
    """
    batch_size, feature_dim = x.shape
    # Norm-center the representations along the batch dimension
    x = x - torch.mean(x, dim=0, keepdim=True)
    cov = torch.matmul(x.T, x) / (batch_size - 1)  # [feature_dim, feature_dim]
    # Target is the identity matrix (no covariance between features)
    id_matrix = torch.eye(feature_dim, device=x.device)
    loss = F.mse_loss(cov, id_matrix)
    return loss
