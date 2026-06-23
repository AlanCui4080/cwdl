from __future__ import annotations

import collections
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_T = 128


class CWModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        V = vocab_size + 1  # index 0 = CTC blank
        self.V = V

        # 2D CNN: (B*K, 1, 16, 128) -> (B*K, 32, 1, 128)
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=(4, 1), padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 1D dilated conv: (B*K, 32, 128) -> (B*K, 128, 128)
        self.conv1d_1 = nn.Conv1d(32, 64, 3, dilation=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1d_2 = nn.Conv1d(64, 64, 3, dilation=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv1d_3 = nn.Conv1d(64, 128, 3, dilation=4, padding=4, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

        # BiLSTM: 超长序列用 LSTM 替代 GRU
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Linear(2 * 256, V)

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
        if K == 0:
            return torch.empty(B, 0, self.V, device=blocks.device)

        x = blocks.reshape(B * K, 1, 16, 128)

        x = self.cnn2d(x)
        x = x.squeeze(2)
        x = self.relu(self.bn1(self.conv1d_1(x)))
        x = self.relu(self.bn2(self.conv1d_2(x)))
        x = self.relu(self.bn3(self.conv1d_3(x)))
        feat = x.transpose(1, 2)
        feat = feat.reshape(B, K * CNN_T, 128)

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

    # log_softmax 在低精度下数值不稳定, 强制 upcast 到 fp32
    logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
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
                s.append(idx2char[i])
            prev = i
        out.append("".join(s))
    return out


# ---------------------------------------------------------------------------

NEG_INF = -float("inf")


def _make_new_beam():
    return collections.defaultdict(lambda: (NEG_INF, NEG_INF))


def _logsumexp(*args) -> float:
    """Numerically stable logsumexp over a variable number of log-probs."""
    m = max(args)
    if m == NEG_INF:
        return NEG_INF
    acc = 0.0
    for a in args:
        acc += math.exp(a - m)
    return m + math.log(acc)


def _prefix_beam_search(log_probs: np.ndarray, beam_size: int = 10,
                        blank: int = 0) -> Tuple[tuple, float]:
    """Run prefix beam search on a single (T x S) log-probability matrix.

    Returns the best label-id prefix (blanks already collapsed) and its
    total log-likelihood (logsumexp of p_blank and p_no_blank).
    """
    T, S = log_probs.shape
    # beam: list of (prefix, (p_blank, p_no_blank)) in log space.
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):
        next_beam = _make_new_beam()
        for s in range(S):
            p = float(log_probs[t, s])
            for prefix, (p_b, p_nb) in beam:
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = _logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = _logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # Repeated char at end: CTC merges, so drop p_nb path.
                    n_p_nb = _logsumexp(n_p_nb, p_b + p)
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                if s == end_t:
                    # Merging case: also keep the unchanged prefix.
                    n_p_b2, n_p_nb2 = next_beam[prefix]
                    n_p_nb2 = _logsumexp(n_p_nb2, p_nb + p)
                    next_beam[prefix] = (n_p_b2, n_p_nb2)

        beam = sorted(next_beam.items(),
                      key=lambda x: _logsumexp(*x[1]),
                      reverse=True)[:beam_size]

    best = beam[0]
    return best[0], _logsumexp(*best[1])


@torch.no_grad()
def prefix_beam_decode(logits: torch.Tensor,
                       input_lengths: torch.Tensor,
                       idx2char: list[str],
                       beam_size: int = 10,
                       blank: int = 0) -> list[str]:
    """CTC prefix beam search decoding for a batch.

    Arguments:
      logits: (B, T, S) raw logits from the model.
      input_lengths: (B,) valid time steps per sample.
      idx2char: vocabulary table; idx2char[0] must be the blank token.
      beam_size: beam width.
      blank: CTC blank index (default 0, matches this model).

    Returns a list of decoded strings (one per batch item).
    """
    # log_softmax in fp32 for numerical stability, matching ctc_loss.
    logp = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
    out: list[str] = []
    for b in range(logp.shape[0]):
        L = int(input_lengths[b].item())
        prefix, _ = _prefix_beam_search(logp[b, :L], beam_size=beam_size,
                                         blank=blank)
        out.append("".join(idx2char[i] for i in prefix))
    return out
