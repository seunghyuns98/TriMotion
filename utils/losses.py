import torch.nn as nn
import torch
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature if temperature is not None else 0.07

    def forward(self, video_embeds, text_embeds):
        logits = torch.matmul(video_embeds, text_embeds.t()) / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        loss = (loss_v2t + loss_t2v) / 2.0

        return loss


