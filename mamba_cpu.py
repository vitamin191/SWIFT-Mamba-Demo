import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()

        # 3. State Space Model (SSM) parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        # x shape: (B, L, D)
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]  
        x = x.transpose(1, 2)
        x = self.activation(x)
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        h = torch.zeros((b, self.d_inner, self.d_state), device=x.device)
        ys = []

        for i in range(l):
            dt_i = dt[:, i, :].unsqueeze(-1)
            x_i = x[:, i, :].unsqueeze(-1)
            B_i = B[:, i, :].unsqueeze(1)
            C_i = C[:, i, :].unsqueeze(1)
            dA = torch.exp(dt_i * A)
            dB = dt_i * B_i
            h = dA * h + dB * x_i
            y_i = (h * C_i).sum(dim=-1)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D
        y = y * self.activation(res)
        out = self.out_proj(y)
        return out