"""Existing spectral TTC calibrator; scheduling and masks live in runner.py."""
import torch
from torch import nn

class FRPlusModule(nn.Module):
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = max(1, freq_bins // groups)
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        B, C, N, T = y_pred.shape
        y = y_pred[:, 0]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)

        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T // 2 + 1 if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)

