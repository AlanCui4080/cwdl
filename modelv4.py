from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_T = 128


class CWModel(nn.Module):

    def __init__(self, vocab_size: int, max_enc_len: int = 20000, max_dec_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size

        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(4, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=(4, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.cnn1d = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, dilation=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=3, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.d_model = 128

        self.register_buffer('encoder_pe', self._sinusoidal_pe(max_enc_len, self.d_model))

        self.decoder_embed = nn.Embedding(vocab_size, self.d_model)
        self.decoder_pe = nn.Embedding(max_dec_len, self.d_model)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            dropout=0.15,
            batch_first=True,
        )

        self.head = nn.Linear(self.d_model, vocab_size)

        self._initialize_weights()

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, blocks: torch.Tensor, num_blocks: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:

        B, K = blocks.shape[:2]
        if K == 0:
            return torch.empty(B, 0, self.vocab_size, device=blocks.device)

        # ===== CNN EXTRACTER =====
        # blocks is (B batchsize, K blockcount, 16 height, 128 length)
        cnn_in = blocks.reshape(B * K, 1, 16, 128)
        # cnn_in is (B*K, 1, 16 height, 128 length)
        cnn_result = self.cnn1d(self.cnn2d(cnn_in).squeeze(2))
        # cnn_result is (B*K, 128 channel, 128 length)
        encoder_in = cnn_result.transpose(1, 2).reshape(B, K * CNN_T, self.d_model)

        # ===== TRANSFOMER ENCODER =====
        # encoder_in is (B, K*128 length, 128 channel)
        encoder_in = encoder_in + self.encoder_pe[:encoder_in.size(1)]
        # encoder_in += position encoding
        encoder_out = self.transformer.encoder(encoder_in)
        
        # ===== TRANSFOMER DECODER =====
        tgt_emb = self.decoder_embed(tgt)
        tgt_emb = tgt_emb + self.decoder_pe(torch.arange(tgt.size(1), device=blocks.device))
        # tgt_emb += position encoding
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(blocks.device)
        # tgt_emb += causal attention mask
        
        decoder_out = self.transformer.decoder(tgt_emb, encoder_out, tgt_mask=tgt_mask, tgt_is_causal=False)
        return self.head(decoder_out)


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0,
            bos_idx: int | None = None, eos_idx: int | None = None,
            penalty: float = 10.0) -> torch.Tensor:
    loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=ignore_index)

    if bos_idx is not None and eos_idx is not None:
        # 可微惩罚: 基于概率而非 argmax, 梯度可回传
        probs = F.softmax(logits, dim=-1)  # (B, L, V)
        B = logits.size(0)
        lengths = (targets != ignore_index).sum(dim=1)  # (B,)

        # 位置0: 惩罚 P(BOS) — 模型不应在首位置预测 BOS
        loss = loss + penalty * probs[:, 0, bos_idx].mean()

        # 末位置: 惩罚 1 - P(EOS) — 模型应在序列末尾预测 EOS
        last_idx = (lengths - 1).clamp(min=0)
        eos_probs = probs[torch.arange(B, device=logits.device), last_idx, eos_idx]
        loss = loss + penalty * (1.0 - eos_probs).mean()

    return loss


@torch.no_grad()
def greedy_decode(model: CWModel, blocks: torch.Tensor, num_blocks: torch.Tensor,
                  bos_idx: int, eos_idx: int, max_len: int = 128) -> list[str]:

    B, K = blocks.shape[:2]
    device = blocks.device

    memory = model.encode_blocks(blocks, num_blocks)

    tgt = torch.full((B, 1), bos_idx, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(B)]

    for _ in range(max_len):
        tgt_emb = model.decoder_embed(tgt) + model.decoder_pe(torch.arange(tgt.size(1), device=device))
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = model.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_is_causal=False)
        logits = model.head(out[:, -1:, :])
        next_token = logits.argmax(-1)

        for b in range(B):
            if not finished[b]:
                tok = next_token[b, 0].item()
                if tok == eos_idx:
                    finished[b] = True
                else:
                    outputs[b].append(tok)

        if finished.all():
            break

        tgt = torch.cat([tgt, next_token], dim=1)

    return outputs
