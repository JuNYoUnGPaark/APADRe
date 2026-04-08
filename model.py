import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union


"""
    - Compute-aware Polynomial Attention Drop-in Replacement(PADRe) architecture
    - Author: JunYoung Park and Myung-Kyu Yi
"""


class PolynomialInteraction(nn.Module):
    def __init__(
      self,
      channels: int, 
      max_degree: int = 3,
      token_kernel: int = 11
    ):
        super().__init__()
        self.max_degree = max_degree
      
        self.channel_mixing = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1) 
            for _ in range(max_degree)
        ])
      
        self.token_mixing = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=token_kernel,
                      padding=token_kernel // 2, groups=channels)
            for _ in range(max_degree)
        ])
      
        self.pre_had_ch = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1)
            for _ in range(max_degree - 1)
        ])
      
        self.pre_had_tok = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=token_kernel,
                      padding=token_kernel // 2, groups=channels)
            for _ in range(max_degree - 1)
        ])

    def forward_first(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_mixing[0](self.channel_mixing[0](x))

    def forward_step(self, z_prev: torch.Tensor, x: torch.Tensor, i: int) -> torch.Tensor:
        y = self.token_mixing[i](self.channel_mixing[i](x))
        z_prev = self.pre_had_tok[i - 1](self.pre_had_ch[i - 1](z_prev))
        return z_prev * y


class ComputeAwareDegreeGate(nn.Module):
    def __init__(
      self,
      channels: int,
      max_degree: int = 3,
      temperature: float = 5.0
    ):
        super().__init__()
        self.max_degree = max_degree
        self.temperature = temperature
        self.use_ste = False
        self.gate = nn.Linear(channels * 3, max_degree)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_mean = x.mean(dim=-1)
        feat_abs  = x.abs().mean(dim=-1)
        feat_flux = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1)
      
        feat = torch.cat([feat_mean, feat_abs, feat_flux], dim=-1)
        logits = self.gate(feat)
        soft_probs = F.softmax(logits / self.temperature, dim=-1)

        if self.training:
            if self.use_ste:
                hard_idx = logits.argmax(dim=-1)
                hard_oh = F.one_hot(hard_idx, self.max_degree).float()
                degree_w = hard_oh - soft_probs.detach() + soft_probs
            else:
                degree_w = soft_probs
        else:
            hard_idx = logits.argmax(dim=-1)
            degree_w = F.one_hot(hard_idx, self.max_degree).float()

        return degree_w, logits, soft_probs


class AdaptivePADReHAR(nn.Module):
    def __init__(
      self,
      input_channels: int,
      hidden_dim: int,
      num_layers: int,
      max_degree: int,
      token_kernel: int,
      dropout: float,
      num_classes: int,
      temperature: float = 5.0
    ):
        super().__init__()
        self.num_layers = num_layers
      
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, 1)
      
        self.gates = nn.ModuleList([
            ComputeAwareDegreeGate(hidden_dim, max_degree, temperature)
            for _ in range(num_layers)
        ])
      
        self.blocks = nn.ModuleList([
            PolynomialInteraction(hidden_dim, max_degree, token_kernel)
            for _ in range(num_layers)
        ])
      
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim  * 2, 1),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(hidden_dim * 2, hidden_dim , 1),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
      
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
      
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
      self, 
      x: torch.Tensor,
      return_gate_info: bool =False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, List[Dict[str, torch.Tensor]]]]
    ]:
        x = self.input_proj(x)
        gate_info_list = []

        for i in range(self.num_layers):
            res = x
            degree_w, logits, soft_probs = self.gates[i](x)
            sel = degree_w.argmax(dim=-1)

            if return_gate_info:
                gate_info_list.append({
                    "soft_probs": soft_probs,
                    "hard_degree": sel + 1,
                })

            if not self.training and x.size(0) == 1:
                target_deg = int(sel.item())
                z = self.blocks[i].forward_first(x)
                for d in range(1, target_deg + 1):
                    z = self.blocks[i].forward_step(z, x, d)
            else:
                max_d = int(sel.max().item())
                outs = []
                z = self.blocks[i].forward_first(x)
                outs.append(z)
              
                for d in range(1, max_d + 1):
                    z = self.blocks[i].forward_step(z, x, d)
                    outs.append(z)

                if self.training and not self.gates[i].use_ste:
                    final_z = torch.zeros_like(outs[0])
                    for d in range(max_d + 1):
                        w = degree_w[:, d].view(-1, 1, 1)
                        final_z += outs[d] * w
                    z = final_z
                else:
                    final_z = torch.zeros_like(outs[0])
                    for d in range(max_d + 1):
                        mask = (sel == d).view(-1, 1, 1).float()
                        final_z += outs[d] * mask
                    z = final_z

            x = self.norms1[i]((z + res).permute(0, 2, 1)).permute(0, 2, 1)
            res2 = x
            x = self.norms2[i]((self.ffns[i](x) + res2).permute(0, 2, 1)).permute(0, 2, 1)

        logits_out = self.classifier(x)

        if not return_gate_info:
            return logits_out

        details = {
          "gate_info": gate_info_list,
        }
        return logits_out, details
