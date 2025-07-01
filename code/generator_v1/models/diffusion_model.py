"""
Diffusion model for protein-conditioned ligand generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DiffusionScheduler:
    """Noise scheduler for diffusion process."""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule: str = "linear"):
        """
        Initialize noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: Noise schedule ('linear', 'cosine')
        """
        self.num_timesteps = num_timesteps
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            # Cosine schedule (often better for generation)
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) - forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Ensure all tensors are on the same device
        device = x_start.device
        t = t.to(device)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        # Ensure all tensors are on the same device
        device = x_t.device
        t = t.to(device)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance."""
        posterior_mean = (
            self.betas[t].reshape(-1, 1) * torch.sqrt(self.alphas_cumprod_prev[t]).reshape(-1, 1) / 
            (1.0 - self.alphas_cumprod[t]).reshape(-1, 1) * x_start +
            self.alphas[t].reshape(-1, 1) * torch.sqrt(1.0 - self.alphas_cumprod_prev[t]).reshape(-1, 1) / 
            (1.0 - self.alphas_cumprod[t]).reshape(-1, 1) * x_t
        )
        
        posterior_variance = self.posterior_variance[t].reshape(-1, 1)
        
        return posterior_mean, posterior_variance
    
    def to(self, device):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class ProteinConditionedUNet(nn.Module):
    """U-Net model for protein-conditioned compound generation."""
    
    def __init__(self,
                 compound_dim: int = 512,
                 protbert_dim: int = 1024,
                 pseq2sites_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initialize protein-conditioned U-Net.
        
        Args:
            compound_dim: Dimension of compound embeddings
            protbert_dim: Dimension of ProtBERT embeddings
            pseq2sites_dim: Dimension of Pseq2Sites embeddings
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.compound_dim = compound_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Protein condition embedding
        self.protein_embed = nn.Sequential(
            nn.Linear(protbert_dim + pseq2sites_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Compound input projection
        self.compound_proj = nn.Linear(compound_dim, hidden_dim)
        
        # Cross-attention layers for protein conditioning
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, dropout) for _ in range(num_layers // 2)
        ])
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, dropout) for _ in range(num_layers // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, compound_dim)
        )
        
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                protein_condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy compound embeddings [batch_size, compound_dim]
            t: Timesteps [batch_size]
            protein_condition: Protein embeddings [batch_size, protbert_dim + pseq2sites_dim]
            
        Returns:
            Predicted noise [batch_size, compound_dim]
        """
        # Embed inputs
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # [batch_size, hidden_dim]
        protein_emb = self.protein_embed(protein_condition)  # [batch_size, hidden_dim]
        x_emb = self.compound_proj(x)  # [batch_size, hidden_dim]
        
        # Add time embedding to compound embedding
        h = x_emb + t_emb
        
        # Apply cross-attention and self-attention layers alternately
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            h = cross_attn(h, protein_emb)
            h = self_attn(h)
            
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for protein conditioning."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_condition = nn.LayerNorm(hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor [batch_size, hidden_dim]
            condition: Key/Value tensor [batch_size, hidden_dim]
            
        Returns:
            Output tensor [batch_size, hidden_dim]
        """
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        condition_seq = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross attention
        x_norm = self.norm_x(x_seq)
        condition_norm = self.norm_condition(condition_seq)
        
        attn_out, _ = self.cross_attention(x_norm, condition_norm, condition_norm)
        x_seq = x_seq + attn_out
        
        # Feed forward
        ffn_out = self.ffn(self.norm_ffn(x_seq))
        x_seq = x_seq + ffn_out
        
        return x_seq.squeeze(1)  # [batch_size, hidden_dim]


class SelfAttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            Output tensor [batch_size, hidden_dim]
        """
        # Add sequence dimension
        x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Self attention
        x_norm = self.norm(x_seq)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # Feed forward
        ffn_out = self.ffn(self.norm_ffn(x_seq))
        x_seq = x_seq + ffn_out
        
        return x_seq.squeeze(1)  # [batch_size, hidden_dim]


class ProteinLigandDiffusion(nn.Module):
    """Complete diffusion model for protein-conditioned ligand generation."""
    
    def __init__(self,
                 compound_dim: int = 512,
                 protbert_dim: int = 1024, 
                 pseq2sites_dim: int = 256,
                 num_timesteps: int = 1000,
                 **model_kwargs):
        """
        Initialize complete diffusion model.
        
        Args:
            compound_dim: Dimension of compound embeddings
            protbert_dim: Dimension of ProtBERT embeddings
            pseq2sites_dim: Dimension of Pseq2Sites embeddings
            num_timesteps: Number of diffusion timesteps
            **model_kwargs: Arguments for U-Net model
        """
        super().__init__()
        
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)
        self.model = ProteinConditionedUNet(
            compound_dim=compound_dim,
            protbert_dim=protbert_dim,
            pseq2sites_dim=pseq2sites_dim,
            **model_kwargs
        )
        
        self.compound_dim = compound_dim
        
    @property
    def num_timesteps(self):
        """Convenience property to access scheduler's num_timesteps."""
        return self.scheduler.num_timesteps
        
    def forward(self, 
                x_start: torch.Tensor,
                protein_condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion pass for training.
        
        Args:
            x_start: Clean compound embeddings [batch_size, compound_dim]
            protein_condition: Protein embeddings [batch_size, protbert_dim + pseq2sites_dim]
            
        Returns:
            Tuple of (noise, predicted_noise, timesteps)
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.scheduler.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t, protein_condition)
        
        return noise, predicted_noise, t
        
    @torch.no_grad()
    def sample(self,
               protein_condition: torch.Tensor,
               initial_compound: Optional[torch.Tensor] = None,
               num_samples: int = 1,
               guidance_scale: float = 1.0,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Sample new compounds given protein condition.
        
        Args:
            protein_condition: Protein embeddings [batch_size, protbert_dim + pseq2sites_dim]
            initial_compound: Initial compound embedding [batch_size, compound_dim] (optional)
            num_samples: Number of samples to generate
            guidance_scale: Guidance scale for conditioning (default: 1.0, no guidance)
            num_inference_steps: Number of denoising steps (default: uses scheduler default)
            
        Returns:
            Generated compound embeddings [batch_size * num_samples, compound_dim]
        """
        batch_size = protein_condition.shape[0]
        device = protein_condition.device
        
        # Use specified inference steps or scheduler default
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        else:
            # Adjust the timestep range if custom inference steps
            num_inference_steps = min(num_inference_steps, self.scheduler.num_timesteps)
        
        # Repeat protein condition for multiple samples
        protein_condition = protein_condition.repeat_interleave(num_samples, dim=0)
        
        # Initialize with noise or given initial compound
        if initial_compound is not None:
            x = initial_compound.repeat_interleave(num_samples, dim=0)
            # Add some noise to initial compound
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        else:
            x = torch.randn(batch_size * num_samples, self.compound_dim, device=device)
            
        # Reverse diffusion with custom inference steps
        step_size = self.scheduler.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[:num_inference_steps]
        
        for t_idx in reversed(timesteps):
            t = torch.full((batch_size * num_samples,), t_idx, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t, protein_condition)
            
            # Apply guidance if specified
            if guidance_scale != 1.0:
                # Simple guidance: amplify the conditioning effect
                uncond_noise = self.model(x, t, torch.zeros_like(protein_condition))
                predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
            
            # Compute previous sample
            alpha_t = self.scheduler.alphas[t_idx]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t_idx]
            alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t_idx]
            beta_t = self.scheduler.betas[t_idx]
            
            # Compute predicted x_0
            pred_x0 = self.scheduler.predict_start_from_noise(x, t, predicted_noise)
            
            # Compute posterior mean
            posterior_mean = (
                beta_t * torch.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod_t) * pred_x0 +
                alpha_t * torch.sqrt(1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x
            )
            
            if t_idx > 0:
                # Add noise for non-final step
                posterior_variance = self.scheduler.posterior_variance[t_idx]
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean
                
        return x
    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler.to(device)
        return self
