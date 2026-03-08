from einops import rearrange
from torch import nn
import torch



class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        B, S, E = x.shape
        return x + self.encoding[:, :S, :]

class AttentionPooling(nn.Module):
    """Pooling spatial avec attention"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x, mask=None):
        # x: (B, C, E), mask: (B, C)
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, E)
        
        # Concat query with input
        x = torch.cat([query, x], dim=1)  # (B, 1+C, E)
        
        if mask is not None:
            # Add False for query position
            query_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([query_mask, mask], dim=1)
        
        x = self.transformer_layer(x, src_key_padding_mask=mask)
        return x[:, 0, :]  # Return query output

class MultiScaleTemporalBlock(nn.Module):
    """
    Capture des patterns temporels à plusieurs échelles:
    - 5-10s: Micro-éveils, mouvements oculaires rapides
    - 30s-1min: Transitions entre stades
    - 2-5min: Cycles de sommeil
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Échelle courte (local): Conv1D avec petit kernel
        self.short_scale = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Échelle moyenne: Conv1D avec kernel moyen
        self.medium_scale = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Échelle longue: Attention
        self.long_scale = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion des échelles
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # x: (B, S, E)
        residual = x
        
        # Short scale
        x_short = self.short_scale(x.transpose(1, 2)).transpose(1, 2)
        
        # Medium scale
        x_medium = self.medium_scale(x.transpose(1, 2)).transpose(1, 2)
        
        # Long scale avec attention
        if mask is not None:
            x_long, _ = self.long_scale(x, x, x, key_padding_mask=mask)
        else:
            x_long, _ = self.long_scale(x, x, x)
        
        # Fusion
        x_fused = torch.cat([x_short, x_medium, x_long], dim=-1)
        x_out = self.fusion(x_fused)
        
        # Residual connection
        x_out = self.norm(x_out + residual)
        
        return x_out

class SleepStagingModel(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=5,
        num_classes=5,
        pooling_head=4,
        dropout=0.1,
        max_seq_length=2160,
    ):
        super().__init__()
        
        if max_seq_length is None:
            max_seq_length = 20000
            
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale blocks
        self.temporal_blocks = nn.ModuleList([
            MultiScaleTemporalBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x, mask):
        B, C, S, E = x.shape
        
        # Spatial pooling
        x = rearrange(x, 'b c s e -> (b s) c e')
        mask_spatial = rearrange(mask[:, :, 0].unsqueeze(1).expand(-1, S, -1), 'b s c -> (b s) c').bool()
        x = self.spatial_pooling(x, mask_spatial)
        x = x.view(B, S, E)
        
        # Temporal modeling
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        
        mask_temporal = mask[:, 0, :].bool()
        for block in self.temporal_blocks:
            x = block(x, mask_temporal)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, mask_temporal