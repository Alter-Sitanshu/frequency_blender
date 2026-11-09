"""Differentiable orthonormal DCT-II. Spatial tensors are RGB in [0, 1]."""
import math
import torch
from torch import nn


class DCT2(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Build angles in float64, then store float32. At 400px, float32 cosine
        # arguments cause avoidable high-frequency error even for constant images.
        n = torch.arange(size, dtype=torch.float64)
        k = n[:, None]
        matrix = torch.cos(math.pi / size * (n + 0.5) * k) * math.sqrt(2 / size)
        matrix[0] /= math.sqrt(2)
        self.register_buffer("matrix", matrix.float())

    def forward(self, x):
        if x.shape[-2:] != self.matrix.shape:
            raise ValueError(f"DCT expects {tuple(self.matrix.shape)}, got {tuple(x.shape[-2:])}")
        # Frequency math remains FP32 even inside detector autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.matrix @ x.float() @ self.matrix.T

    def inverse(self, coefficients):
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            return self.matrix.T @ coefficients.float() @ self.matrix


def frequency_priors(size, device=None):
    """Paper Sec. 3: x+y <= 1/16, (1/16,1/2], >1/2."""
    axis = torch.linspace(0, 1, size, device=device)
    radius = axis[:, None] + axis[None, :]
    return torch.stack((radius <= 1/16, (radius > 1/16) & (radius <= 1/2),
                        radius > 1/2)).float()[None]


class FPNet(nn.Module):
    """Four stride-2 convs, three independent four-stage PixelShuffle decoders.

    Widths/activations are implementation choices (not specified in supplied PDF).
    Independent sigmoid masks retain the paper's nontrivial integrity loss.
    """
    def __init__(self, size=224, width=16):
        super().__init__()
        if size < 32 or size % 16:
            raise ValueError("FPNet size must be >=32 and divisible by 16 (400 is valid).")
        self.size, self.width = size, width
        self.dct = DCT2(size)
        widths = [3, width, 2*width, 4*width, 8*width]
        self.encoder = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(a, b, 3, 2, 1), nn.LeakyReLU(0.2))
            for a, b in zip(widths[:-1], widths[1:])])
        self.decoders = nn.ModuleList()
        for _ in range(3):
            outputs = [4*width, 2*width, width, 1]
            inputs = [8*width] + outputs[:-1]
            layers = []
            for i, (a, b) in enumerate(zip(inputs, outputs)):
                layers += [nn.Conv2d(a, b*4, 3, 1, 1), nn.PixelShuffle(2)]
                layers += [nn.Sigmoid() if i == 3 else nn.LeakyReLU(0.2)]
            self.decoders.append(nn.Sequential(*layers))
        self.register_buffer("prior", frequency_priors(size))

    def forward(self, x):
        coefficients = self.dct(x)
        hidden = self.encoder(coefficients)
        masks = torch.cat([decoder(hidden) for decoder in self.decoders], dim=1)
        return coefficients, masks

    def components(self, x):
        coefficients, masks = self(x)
        return coefficients[:, None] * masks[:, :, None], masks

    def blend(self, real, fake, clamp=True):
        """Eq. 6: ALL masks predicted from fake, applied to the correct donor.

        The prose around Eq. 6 swaps xr/xf; donor roles follow Sec. 4.3 intent.
        """
        fake_c, masks = self(fake)
        real_c = self.dct(real)
        blended = self.dct.inverse(real_c * (masks[:, 0:1] + masks[:, 2:3])
                                   + fake_c * masks[:, 1:2])
        return blended.clamp(0, 1) if clamp else blended
