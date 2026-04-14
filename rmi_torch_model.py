from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101


def _make_spatial_features(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y_indices = torch.arange(height, device=device, dtype=torch.float32)
    x_indices = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing="ij")

    xmin = grid_x / width * 2.0 - 1.0
    xmax = (grid_x + 1.0) / width * 2.0 - 1.0
    ymin = grid_y / height * 2.0 - 1.0
    ymax = (grid_y + 1.0) / height * 2.0 - 1.0
    xctr = (xmin + xmax) / 2.0
    yctr = (ymin + ymax) / 2.0

    spatial = torch.stack(
        [xmin, ymin, xmax, ymax, xctr, yctr, torch.full_like(xctr, 1.0 / width), torch.full_like(yctr, 1.0 / height)],
        dim=-1,
    )
    return spatial.unsqueeze(0).repeat(batch_size, 1, 1, 1)


class ResNetBackbone(nn.Module):
    def __init__(self, weights_name: str):
        super().__init__()
        weights = getattr(ResNet101_Weights, weights_name)
        backbone = resnet101(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.stem(image)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        return feat


class RMIResNetModel(nn.Module):
    def __init__(self, config: Dict, vocab_size: int):
        super().__init__()
        model_cfg = config["model"]
        self.input_size = int(model_cfg["input_size"])
        self.num_steps = int(model_cfg["num_steps"])
        self.visual_dim = int(model_cfg["visual_dim"])
        self.word_embed_dim = int(model_cfg["word_embed_dim"])
        self.rnn_hidden_dim = int(model_cfg["rnn_hidden_dim"])
        self.mlp_dim = int(model_cfg["mlp_dim"])

        self.backbone = ResNetBackbone(model_cfg["backbone_weights"])
        self.embedding = nn.Embedding(vocab_size, self.word_embed_dim, padding_idx=0)
        self.word_lstm = nn.LSTMCell(self.word_embed_dim, self.rnn_hidden_dim)
        self.action_lstm = nn.LSTMCell(self.visual_dim + self.word_embed_dim + self.rnn_hidden_dim + 8, self.mlp_dim)
        self.prediction = nn.Conv2d(self.mlp_dim, 1, kernel_size=1, stride=1)

    def forward(self, image: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        visual_feat = self.backbone(image)
        batch_size, channels, height, width = visual_feat.shape
        visual_feat = F.normalize(visual_feat, dim=1)
        spatial = _make_spatial_features(batch_size, height, width, image.device).permute(0, 3, 1, 2)
        spatial = spatial.contiguous()

        embedded_words = self.embedding(words)
        h_w = image.new_zeros((batch_size, self.rnn_hidden_dim))
        c_w = image.new_zeros((batch_size, self.rnn_hidden_dim))
        h_a = image.new_zeros((batch_size * height * width, self.mlp_dim))
        c_a = image.new_zeros((batch_size * height * width, self.mlp_dim))

        for step in range(self.num_steps):
            token = words[:, step]
            non_pad = token.ne(0)
            if not bool(non_pad.any()):
                continue

            word_feat = embedded_words[:, step, :]
            h_w_new, c_w_new = self.word_lstm(word_feat, (h_w, c_w))
            word_mask = non_pad.unsqueeze(1)
            h_w = torch.where(word_mask, h_w_new, h_w)
            c_w = torch.where(word_mask, c_w_new, c_w)

            lang_feat = F.normalize(h_w, dim=1).view(batch_size, self.rnn_hidden_dim, 1, 1).expand(-1, -1, height, width)
            current_word = word_feat.view(batch_size, self.word_embed_dim, 1, 1).expand(-1, -1, height, width)
            fused = torch.cat([visual_feat, current_word, lang_feat, spatial], dim=1)
            fused = fused.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, -1)

            h_a_new, c_a_new = self.action_lstm(fused, (h_a, c_a))
            action_mask = non_pad.view(batch_size, 1, 1, 1).expand(-1, height, width, 1).reshape(batch_size * height * width, 1)
            h_a = torch.where(action_mask, h_a_new, h_a)
            c_a = torch.where(action_mask, c_a_new, c_a)

        lstm_output = h_a.view(batch_size, height, width, self.mlp_dim).permute(0, 3, 1, 2)
        eps = 1e-3
        lstm_output = torch.clamp(lstm_output, min=-1.0 + eps, max=1.0 - eps)
        lstm_output = 0.5 * (torch.log1p(eps + lstm_output) - torch.log1p(eps - lstm_output))
        logits = self.prediction(F.relu(lstm_output))
        return F.interpolate(logits, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
