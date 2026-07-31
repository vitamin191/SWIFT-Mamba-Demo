import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba
from mamba_cpu import Mamba

ABLATION_CHOICES = (
    "cnn_tanh",
    "cnn_softclip",
    "time_only_softclip",
    "freq_only_softclip",
    "dual_unbounded",
    "dual_hardclip",
    "dual_tanh",
    "dual_softclip",
)


class BaselineCNNBlock(nn.Module):
    """Two-layer residual CNN block used as the non-SSM baseline."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class PhysicalMambaBlock(nn.Module):
    """
    Dual-path physical block.

    Time path:
        Mamba scans along the time-frame dimension W.

    Frequency path:
        Depthwise separable convolution scans locally along H.

    The two paths can be independently disabled for ablation.
    """

    def __init__(
        self,
        in_channels: int,
        h_feat: int,
        use_time: bool = True,
        use_freq: bool = True,
        d_state: int = 16,
        expand: int = 2,
    ):
        super().__init__()

        if not use_time and not use_freq:
            raise ValueError("At least one of use_time/use_freq must be True.")

        self.use_time = use_time
        self.use_freq = use_freq
        self.in_channels = in_channels
        self.h_feat = h_feat

        if self.use_time:
            self.d_time = in_channels * h_feat
            self.mamba_time = Mamba(
                d_model=self.d_time,
                d_state=d_state,
                d_conv=4,
                expand=expand,
            )
            self.norm_time = nn.LayerNorm(self.d_time)

        if self.use_freq:
            self.freq_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(5, 1),
                    padding=(2, 0),
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            )

        if self.use_time and self.use_freq:
            self.fusion = nn.Conv2d(
                in_channels * 2,
                in_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.fusion = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if h != self.h_feat:
            raise RuntimeError(
                f"PhysicalMambaBlock expected H={self.h_feat}, but received H={h}. "
                "Check the STFT size and encoder downsampling."
            )

        outputs = []

        if self.use_time:
            x_time = x.reshape(b, c * h, w).transpose(1, 2)
            x_time = self.norm_time(self.mamba_time(x_time) + x_time)
            out_time = x_time.transpose(1, 2).reshape(b, c, h, w)
            outputs.append(out_time)

        if self.use_freq:
            outputs.append(self.freq_conv(x))

        if len(outputs) == 2:
            out = self.fusion(torch.cat(outputs, dim=1))
        else:
            out = outputs[0]

        return x + out


class SWIFTMambaNet(nn.Module):
    """
    SWIFT-Mamba with unified ablation switches.

    Modes
    -----
    cnn_tanh:
        CNN bottleneck + tanh mask.
    cnn_softclip:
        CNN bottleneck + proposed soft-clipping mask.
    time_only_softclip:
        Time-scan Mamba only + proposed soft-clipping mask.
    freq_only_softclip:
        Frequency convolution only + proposed soft-clipping mask.
    dual_unbounded:
        Full dual-path block + unbounded linear output.
    dual_hardclip:
        Full dual-path block + torch.clamp output.
    dual_tanh:
        Full dual-path block + scaled tanh output.
    dual_softclip:
        Full proposed model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 16,
        ablation_mode: str = "dual_softclip",
        mask_bound: float = 10.0,
    ):
        super().__init__()

        if ablation_mode not in ABLATION_CHOICES:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode!r}. "
                f"Available modes: {ABLATION_CHOICES}"
            )

        self.ablation_mode = ablation_mode
        self.mask_bound = float(mask_bound)

        self.enc1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_filters,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_filters),
            nn.SiLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(
                base_filters,
                base_filters * 2,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_filters * 2),
            nn.SiLU(),
        )

        bottleneck_channels = base_filters * 2

        if ablation_mode.startswith("cnn_"):
            block_factory = lambda: BaselineCNNBlock(bottleneck_channels)
        elif ablation_mode == "time_only_softclip":
            block_factory = lambda: PhysicalMambaBlock(
                bottleneck_channels,
                h_feat=16,
                use_time=True,
                use_freq=False,
            )
        elif ablation_mode == "freq_only_softclip":
            block_factory = lambda: PhysicalMambaBlock(
                bottleneck_channels,
                h_feat=16,
                use_time=False,
                use_freq=True,
            )
        else:
            block_factory = lambda: PhysicalMambaBlock(
                bottleneck_channels,
                h_feat=16,
                use_time=True,
                use_freq=True,
            )

        self.block1 = block_factory()
        self.block2 = block_factory()

        self.dec2 = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                base_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_filters),
            nn.SiLU(),
        )

        self.skip_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.dec1 = nn.Conv2d(
            base_filters,
            2,
            kernel_size=3,
            padding=1,
        )

    def _apply_mask_activation(self, logits: torch.Tensor) -> torch.Tensor:
        mode = self.ablation_mode
        bound = self.mask_bound

        if mode == "dual_unbounded":
            return logits

        if mode == "dual_hardclip":
            return torch.clamp(logits, min=-bound, max=bound)

        if mode in {"cnn_tanh", "dual_tanh"}:
            return bound * torch.tanh(logits)

        # cnn_softclip, time_only_softclip, freq_only_softclip, dual_softclip
        return bound * logits / (1.0 + torch.abs(logits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.enc1(x)
        feat2 = self.enc2(feat1)

        feat_mid = self.block1(feat2)
        feat_mid = self.block2(feat_mid)

        up2 = F.interpolate(
            feat_mid,
            scale_factor=(2, 1),
            mode="bilinear",
            align_corners=False,
        )
        out_dec2 = self.dec2(up2) + self.skip_alpha * feat1

        up1 = F.interpolate(
            out_dec2,
            scale_factor=(2, 1),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.dec1(up1)

        return self._apply_mask_activation(logits)


if __name__ == "__main__":
    # Lightweight shape sanity check.
    x = torch.randn(2, 3, 64, 32)
    for mode in ABLATION_CHOICES:
        model = SWIFTMambaNet(ablation_mode=mode)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{mode:24s} output={tuple(y.shape)}, params={params:,}")
