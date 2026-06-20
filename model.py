

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
            nn.Conv2d(1, 16, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1d_1 = nn.Conv1d(32, 64, 3, dilation=1, padding=1, bias=False)
        self.conv1d_2 = nn.Conv1d(64, 64, 3, dilation=2, padding=2, bias=False)
        self.conv1d_3 = nn.Conv1d(64, 64, 3, dilation=4, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Linear(2 * 64, V)

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
        x = blocks.reshape(B * K, 1, 16, 128)

        x = self.cnn(x)
        x = x.squeeze(2)
        x = self.relu(self.bn1(self.conv1d_1(x)))
        x = self.relu(self.bn2(self.conv1d_2(x)))
        x = self.relu(self.bn3(self.conv1d_3(x)))
        feat = x.transpose(1, 2)
        feat = feat.reshape(B, K * CNN_T, 64)

        lengths = (num_blocks * CNN_T).long()

        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.bigru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=K * CNN_T)
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
