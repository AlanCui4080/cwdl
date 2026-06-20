from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_T = 128


class CWModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        V = vocab_size + 1
        self.V = V

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.max_len = 8192
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, 128))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.3, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=12)

        self.head = nn.Linear(128, V)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, blocks: torch.Tensor,
                num_blocks: torch.Tensor) -> torch.Tensor:
        B, K = blocks.shape[:2]
        T = K * CNN_T
        x = blocks.reshape(B * K, 1, 16, 128)

        x = self.cnn(x)
        x = x.squeeze(2)
        feat = x.transpose(1, 2)
        feat = feat.reshape(B, T, 128)

        memory = feat + self.pos[:, :T, :]

        lengths = (num_blocks * CNN_T).long()
        pad_mask = torch.arange(T, device=blocks.device).unsqueeze(0) >= lengths.unsqueeze(1)

        out = self.encoder(memory, src_key_padding_mask=pad_mask)
        logits = self.head(out)
        return logits


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor,
             input_lengths: torch.Tensor, target_lengths: torch.Tensor):
    logp = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(logp, targets, input_lengths, target_lengths,
                      blank=0, reduction="mean", zero_infinity=True)


@torch.no_grad()
def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor,
                  idx2char: list[str]) -> list[str]:
    pred = logits.argmax(-1)
    out: list[str] = []
    for b in range(logits.size(0)):
        L = int(input_lengths[b].item())
        ids = pred[b, :L].tolist()
        s: list[str] = []
        prev = -1
        for i in ids:
            if i != prev and i != 0:
                s.append(idx2char[i - 1])
            prev = i
        out.append("".join(s))
    return out
