import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class VAE_Projector(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        latent_channels=16,
        output_dim=768,
        hidden_dim=512,
        num_layers=2,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.spatial_cls_token, std=0.02)

        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.temporal_cls_token, std=0.02)

        self.spatial_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latents, return_frame_embeds=False):
        B, C, T, H, W = latents.shape

        x = self.feature_extractor(latents)

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        x_spatial = x.view(B * T, H * W, self.hidden_dim)

        spatial_cls = self.spatial_cls_token.expand(B * T, -1, -1)
        spatial_sequence = torch.cat([spatial_cls, x_spatial], dim=1)

        spatial_encoded = self.spatial_encoder(spatial_sequence)

        frame_embeds = spatial_encoded[:, 0]
        frame_embeds = frame_embeds.view(B, T, self.hidden_dim)

        temporal_cls = self.temporal_cls_token.expand(B, -1, -1)
        temporal_sequence = torch.cat([temporal_cls, frame_embeds], dim=1)

        temporal_encoded = self.temporal_encoder(temporal_sequence)

        embedding = temporal_encoded

        embedding = self.final_norm(embedding)
        embedding = self.dropout(embedding)
        embedding = self.final_proj(embedding)

        video_global_embeds = F.normalize(embedding[:, 0], dim=-1)
        video_frame_embeds = F.normalize(embedding[:, 1:], dim=-1)

        return {
            "video_global_embeds": video_global_embeds,
            "video_frame_embeds": video_frame_embeds,
        }

