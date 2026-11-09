"""An explicitly approximate spatial self-blender, not official SBI."""
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


@torch.no_grad()
def spatial_self_blend(real):
    batch, _, height, width = real.shape
    kwargs = dict(device=real.device, dtype=real.dtype)
    theta = torch.eye(2, 3, **kwargs)[None].repeat(batch, 1, 1)
    theta[:, 0, 0] = 1 + (torch.rand(batch, **kwargs)-.5)*.12
    theta[:, 1, 1] = 1 + (torch.rand(batch, **kwargs)-.5)*.12
    theta[:, :, 2] = (torch.rand(batch, 2, **kwargs)-.5)*.08
    warped = F.grid_sample(real, F.affine_grid(theta, real.shape, align_corners=False),
                           padding_mode="reflection", align_corners=False)
    contrast = .8 + .4*torch.rand(batch, 3, 1, 1, **kwargs)
    bias = (torch.rand(batch, 3, 1, 1, **kwargs)-.5)*.1
    warped = (warped*contrast + bias).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, **kwargs),
                           torch.linspace(-1, 1, width, **kwargs), indexing="ij")
    cx, cy = (torch.rand(batch, 2, 1, 1, **kwargs)-.5).unbind(1)
    rx = .55 + .2*torch.rand(batch, 1, 1, **kwargs)
    ry = .65 + .2*torch.rand(batch, 1, 1, **kwargs)
    ellipse = (((xx-.15*cx)/rx)**2 + ((yy-.15*cy)/ry)**2 < 1).float()[:, None]
    mask = gaussian_blur(ellipse, [15, 15], [3., 3.])
    mask *= .5 + .5*torch.rand(batch, 1, 1, 1, **kwargs)
    return (real*(1-mask) + warped*mask).clamp(0, 1)


@torch.no_grad()
def training_pair(real, parser=None, blend_probability=.2):
    fake = spatial_self_blend(real)
    if parser is not None and blend_probability > 0:
        selected = torch.rand(len(real), device=real.device) < blend_probability
        if selected.any():
            fake[selected] = parser.blend(real[selected], fake[selected])
    return fake
