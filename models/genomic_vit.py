"""
Vision Transformer for Genomic Sequences

Predicts drug response from DNA sequences using transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GenomicPatchEmbedding(nn.Module):
    """
    Split DNA sequence into patches and embed them
    
    Example:
        Input: (batch, 1024, 4)  # one-hot encoded sequence
        Output: (batch, 64, 256)  # 64 patches, each 256-dim
    """
    
    def __init__(self, seq_length=1024, patch_size=16, in_channels=4, embed_dim=256):
        """
        arguments:
            seq_length: Length of DNA sequence (1024 bases)
            patch_size: Number of bases per patch (16)
            in_channels: Input channels (4 for ATCG one-hot)
            embed_dim: Embedding dimension (256)
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size  # 1024 / 16 = 64
        
        # Linear projection of patches
        # Input: (batch, seq_length, in_channels)
        # Reshape to: (batch, num_patches, patch_size * in_channels)
        # Project to: (batch, num_patches, embed_dim)
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)
    
    def forward(self, x):
        """
        arguments:
            x: (batch, seq_length, in_channels) - one-hot encoded
        
        returns:
            (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape into patches
        # (batch, seq_length, in_channels) -> (batch, num_patches, patch_size, in_channels)
        x = x.view(batch_size, self.num_patches, self.patch_size, -1)
        
        # Flatten patches
        # (batch, num_patches, patch_size, in_channels) -> (batch, num_patches, patch_size * in_channels)
        x = x.view(batch_size, self.num_patches, -1)
        
        # Project to embed_dim
        x = self.projection(x)  # (batch, num_patches, embed_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Add positional information to patch embeddings
    
    Why important for DNA:
    - Same sequence at different positions can have different functions
    - Promoter regions vs coding regions vs regulatory elements
    """
    
    def __init__(self, embed_dim, max_len=64):
        """
        Args:
            embed_dim: Embedding dimension (256)
            max_len: Maximum number of patches (64)
        """
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(1, max_len, embed_dim)  # (1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        argumnrnts:
            x: (batch, num_patches, embed_dim)
        
        Returns:
            (batch, num_patches, embed_dim) with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with multi-head attention
    """
    
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        """
        arguments:
            embed_dim: Embedding dimension (256)
            num_heads: Number of attention heads (8)
            mlp_dim: Hidden dimension of feed-forward network (1024)
            dropout: Dropout rate (0.1)
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        arguments:
            x: (batch, num_patches, embed_dim)
        
        returns:
            (batch, num_patches, embed_dim)
        """
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    stack of transformer encoder layers
    """
    
    def __init__(self, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1):
        """
        arguments:
            embed_dim: Embedding dimension (256)
            num_layers: Number of encoder layers (8)
            num_heads: Number of attention heads (8)
            mlp_dim: Hidden dimension of FFN (1024)
            dropout: Dropout rate (0.1)
        """
        super().__init__()
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        arguments:
            x: (batch, num_patches, embed_dim)
        
        returns:
            (batch, num_patches, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class GenomicClassificationHead(nn.Module):
    """
    Classification head for drug response prediction
    
    Outputs:
    - drug_response: Main prediction (0-1)
    - uncertainty: Confidence estimate
    """
    
    def __init__(self, embed_dim, num_classes=1):
        """
        argumnents:
            embed_dim: Input dimension (256)
            num_classes: Number of output classes (1 for regression)
        """
        super().__init__()
        
        # Main prediction head
        self.fc_response = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Output between 0-1
        )
        
        # Uncertainty estimation head
        self.fc_uncertainty = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softplus()  # Always positive
        )
    
    def forward(self, x):
        """
        arguments:
            x: (batch, embed_dim) - pooled features
        
        returns:
            drug_response: (batch, 1)
            uncertainty: (batch, 1)
        """
        drug_response = self.fc_response(x)
        uncertainty = self.fc_uncertainty(x)
        
        return drug_response, uncertainty


class GenomicViT(nn.Module):
    """
    Complete Vision Transformer for Genomic Sequences
    
    Predicts drug response from DNA sequences
    """
    
    def __init__(
        self,
        seq_length=1024,
        patch_size=16,
        in_channels=4,
        embed_dim=256,
        num_layers=8,
        num_heads=8,
        mlp_dim=1024,
        num_classes=1,
        dropout=0.1
    ):
        """
        arguments:
            seq_length: DNA sequence length (1024)
            patch_size: Bases per patch (16)
            in_channels: Input channels (4 for ATCG)
            embed_dim: Embedding dimension (256)
            num_layers: Number of transformer layers (8)
            num_heads: Number of attention heads (8)
            mlp_dim: FFN hidden dimension (1024)
            num_classes: Output dimension (1)
            dropout: Dropout rate (0.1)
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Components
        self.patch_embed = GenomicPatchEmbedding(
            seq_length=seq_length,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=seq_length // patch_size
        )
        
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        self.classification_head = GenomicClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        arguments:
            x: (batch, seq_length, 4) - one-hot encoded DNA
        
        returns:
            drug_response: (batch, 1) - prediction
            uncertainty: (batch, 1) - confidence
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)  # (batch, num_patches, embed_dim)
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, num_patches, embed_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Classification
        drug_response, uncertainty = self.classification_head(x)
        
        return drug_response, uncertainty
    
    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization
        
        arguments:
            x: (batch, seq_length, 4)
        
        returns:
            List of attention maps from each layer
        """
        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Collect attention weights
        attention_maps = []
        
        for layer in self.transformer.layers:
            # Get attention weights
            _, attn_weights = layer.attention(x, x, x, need_weights=True)
            attention_maps.append(attn_weights)
            
            # Continue forward pass
            x = layer(x)
        
        return attention_maps

