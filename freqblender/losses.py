import torch
from torch.nn import functional as F
from .models import binary_margin


def parser_loss(parser, real, fake, identity, discriminator,
                weights=(1/12, 1., 1e-3, 1/4)):
    """Eqs. 1-5. Pixel/feature reductions are means, a documented adaptation.

    Do not wrap identity/discriminator forward passes in no_grad: FPNet needs
    input gradients through the frozen auxiliaries.
    """
    both = torch.cat([real, fake])
    parts, masks = parser.components(both)
    semantic = parser.dct.inverse(parts[:, 0])
    clean = parser.dct.inverse(parts[:, 0] + parts[:, 1])
    with torch.no_grad():
        target_identity = identity(both)
    ff = F.mse_loss(identity(semantic), target_identity)
    qa = F.mse_loss(clean, both)
    pi = sum(F.mse_loss(masks[:, i], parser.prior[:, i].expand_as(masks[:, i]))
             for i in range(3)) + (masks.sum(1)-1).square().mean()
    r, f = parts.chunk(2)
    real_combinations = [r[:, 0], f[:, 0], r[:, 0]+r[:, 1]]
    fake_combinations = [f[:, 0]+f[:, 1], r[:, 0]+f[:, 1]]
    # Separate set means reproduce the 1/|Cr| and 1/|Cf| weighting.
    cr = parser.dct.inverse(torch.cat(real_combinations))
    cf = parser.dct.inverse(torch.cat(fake_combinations))
    ad = F.cross_entropy(discriminator(cr), torch.zeros(len(cr), dtype=torch.long, device=real.device))
    ad += F.cross_entropy(discriminator(cf), torch.ones(len(cf), dtype=torch.long, device=real.device))
    terms = {"ff": ff, "ad": ad, "qa": qa, "pi": pi}
    total = sum(weight * value for weight, value in zip(weights, terms.values()))
    return total, terms


def distillation_kl(student_logits, teacher_logits, temperature=4.):
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits.detach() / temperature, dim=1),
                    reduction="batchmean") * temperature**2


@torch.no_grad()
def counterfactual_views(x, donor, parser, strength=.25, epsilon=.02, max_gain=4.):
    """Same base image; separate structural/noise coefficient interventions.

    Bound the pixel RMS to epsilon before clipping. Normalize the three sigmoid
    masks ONLY here so the proposed intervention has a partition of unity.
    """
    coefficients, masks = parser(x)
    masks = masks / masks.sum(1, keepdim=True).clamp_min(1e-6)
    donor_coefficients = parser.dct(donor)
    outputs = []
    for index in (1, 2):
        delta = parser.dct.inverse((donor_coefficients-coefficients)*masks[:, index:index+1])
        rms = delta.square().mean((1, 2, 3), keepdim=True).sqrt()
        gain = (epsilon/rms.clamp_min(1e-8)).clamp(max=max_gain)
        outputs.append((x + strength*gain*delta).clamp(0, 1))
    return tuple(outputs)


def secant_loss(student_base, student_struct, student_noise,
                teacher_base, teacher_struct, teacher_noise, labels,
                x, x_struct, x_noise, noise_weight=.25, energy_floor=1e-3):
    """Proposed structural secant transfer with teacher-gated noise invariance.

    The RMS-normalized logit response approximates a directional derivative;
    clipping is accounted for by measuring the ACTUAL image displacement.
    """
    teacher_base = teacher_base.detach()
    confidence = teacher_base.softmax(1).gather(1, labels[:, None]).squeeze(1)
    gate = (2*confidence - 1).clamp(min=0).detach()
    def response(base, view, altered):
        distance = (altered-x).square().mean((1, 2, 3)).sqrt()
        # 0.01 keeps derivative targets in numerically tractable logit units.
        return .01*(binary_margin(view)-binary_margin(base))/distance.clamp_min(energy_floor)
    s_struct = response(student_base, student_struct, x_struct)
    t_struct = response(teacher_base, teacher_struct.detach(), x_struct)
    structure = F.smooth_l1_loss(s_struct, t_struct, reduction="none")
    t_noise = response(teacher_base, teacher_noise.detach(), x_noise)
    s_noise = response(student_base, student_noise, x_noise)
    noise_gate = torch.exp(-t_noise.abs()).detach()
    noise = F.smooth_l1_loss(s_noise, torch.zeros_like(s_noise), reduction="none")
    # Mean (not sum/gate.sum): unreliable teachers reduce the total influence.
    return (gate*structure).mean() + noise_weight*(gate*noise_gate*noise).mean()
