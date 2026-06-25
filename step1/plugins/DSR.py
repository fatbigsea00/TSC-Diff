# Dual-space Frequency Distribution Broadening (DFDB)
# ======================================================
# A dual-space frequency distribution broadening framework that, through frequency-domain
# amplitude perturbation in both pixel space and latent space, broadens the feature
# distribution in few-shot sonar image training, improving the robustness and generalization
# of the diffusion model.
#
# Contains two complementary modules:
#   PFA (Pixel-level Frequency Augmentation)  — pixel-level frequency augmentation (formerly DoublePerturb)
#   LFA (Latent-level Frequency Augmentation) — latent-space frequency augmentation (after VAE encoding)

import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class RandomConv(nn.Module):
    """
    Random convolution module: generates a random convolution kernel to apply global perturbation to features.
    """
    def __init__(self, in_channels=3, kernel_size=3, std=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.std = std
        self.padding = kernel_size // 2

    def forward(self, x):
        weight = (torch.randn(
            self.in_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            device=x.device
        ) * self.std)

        x = F.conv2d(x, weight, padding=self.padding, bias=None)
        return x


class MaskDilator(nn.Module):
    """
    Mask dilation module: expands the foreground mask region to ensure the integrity of the foreground area.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, mask):
        new_mask = mask.float().unsqueeze(1)
        new_mask = F.max_pool2d(
            new_mask,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1
        )
        new_mask = F.avg_pool2d(
            new_mask,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1
        )
        new_mask = new_mask.squeeze(1)
        return new_mask


# =====================================================================
# PFA: Pixel-level Frequency Augmentation (formerly Dual Style Randomization)
# =====================================================================
class DoublePerturb(nn.Module):
    """
    Pixel-level Frequency Augmentation (PFA)

    Applies frequency-domain amplitude perturbation to the RGB pixel image before VAE encoding:
      - Foreground path: borrows amplitude from a randomly selected superpixel region and
        blends it with the amplitude of the foreground region.
      - Global path: generates a substitute amplitude via random convolution and fuses it
        with the original amplitude.
    The phase is preserved (structure unchanged) and only the amplitude is perturbed (broadening
    the low-level texture distribution), thereby enriching pixel-level feature diversity under
    few-shot conditions and reducing the risk of overfitting.
    """
    def __init__(self, feat_dim=3, random_size=3, random_std=0.1, prob=1.0, mask_size=9):
        super(DoublePerturb, self).__init__()
        self.random_conv = RandomConv(feat_dim, random_size, random_std)
        self.mask_dilator = MaskDilator(kernel_size=mask_size)

        self.feat_dim = feat_dim
        self.prob = prob

    def forward(self, feature, mask, spixel_mask):
        if random.random() < self.prob:
            feature = self.foreground_perturb(feature, mask, spixel_mask)

        if random.random() < self.prob:
            feature = self.global_perturb(feature)

        return feature

    def foreground_perturb(self, feature, mask, spixel_mask):
        bs, nc, fh, fw = feature.shape
        mask = F.interpolate(mask.unsqueeze(1).float(), size=feature.shape[-2:], mode='nearest').squeeze(1)
        dilate_mask = self.mask_dilator(mask)
        spixel_mask = F.interpolate(spixel_mask.float(), size=feature.shape[-2:], mode='nearest')

        perturb_feature_list = []
        for epi in range(bs):
            cur_feat = feature[epi]
            cur_dilate_mask = dilate_mask[epi]
            cur_spixel_mask = spixel_mask[epi][0]

            unique_labels = torch.unique(cur_spixel_mask)
            selected_label = np.random.choice(unique_labels.cpu().numpy())
            selected_mask = torch.zeros_like(cur_spixel_mask)
            selected_mask[cur_spixel_mask == selected_label] = 1.0

            M = selected_mask.sum()
            if M < (fh * fw) / unique_labels.shape[0] or cur_dilate_mask.sum() < (fh * fw) / unique_labels.shape[0]:
                perturb_feature_list.append(cur_feat.view(1, nc, fh, fw))
                continue

            coords1 = torch.nonzero(cur_dilate_mask)
            coords2 = torch.nonzero(selected_mask)
            y1_min, x1_min = torch.min(coords1, dim=0).values
            y1_max, x1_max = torch.max(coords1, dim=0).values
            y2_min, x2_min = torch.min(coords2, dim=0).values
            y2_max, x2_max = torch.max(coords2, dim=0).values

            patch1 = cur_feat[:, y1_min:y1_max + 1, x1_min:x1_max + 1]
            patch2 = cur_feat[:, y2_min:y2_max + 1, x2_min:x2_max + 1]
            patch2 = F.interpolate(patch2.unsqueeze(0), size=patch1.shape[-2:], mode='bilinear',
                                   align_corners=True).squeeze(0)

            perturb_weight = (0.25 * torch.randn_like(patch1.mean(dim=(1, 2), keepdim=True))).unsqueeze(0).clamp(-1.0,
                                                                                                                 1.0)

            feature_fg_freq = torch.fft.fftshift(torch.fft.fft2(patch1.unsqueeze(0)))
            phase = torch.angle(feature_fg_freq)
            feature_bg_freq = torch.fft.fftshift(torch.fft.fft2(patch2.unsqueeze(0)))
            bg_amplitude = torch.abs(feature_bg_freq)
            amplitude = perturb_weight * bg_amplitude + (1 - perturb_weight) * torch.abs(feature_fg_freq)

            fusion_freq = torch.polar(amplitude, phase)
            patch1 = torch.fft.ifft2(torch.fft.ifftshift(fusion_freq)).real

            perturb_feat = cur_feat.clone()
            perturb_feat[:, y1_min:y1_max + 1, x1_min:x1_max + 1] = patch1
            perturb_feat = perturb_feat * cur_dilate_mask.unsqueeze(0) + cur_feat * (1 - cur_dilate_mask.unsqueeze(0))

            perturb_feature_list.append(perturb_feat.view(1, nc, fh, fw))

        perturb_feature = torch.cat(perturb_feature_list, dim=0)

        return perturb_feature

    def global_perturb(self, feature):
        random_feature = self.random_conv(feature)

        feature_freq = torch.fft.fftshift(torch.fft.fft2(feature))
        phase = torch.angle(feature_freq)
        random_feature_freq = torch.fft.fftshift(torch.fft.fft2(random_feature))
        amplitude = torch.abs(random_feature_freq)

        fusion_freq = torch.polar(amplitude, phase)
        perturb_feature = torch.fft.ifft2(torch.fft.ifftshift(fusion_freq)).real

        return perturb_feature


# =====================================================================
# LFA: Latent-level Frequency Augmentation
# =====================================================================
class LatentFrequencyPerturb(nn.Module):
    """
    Latent-level Frequency Augmentation (LFA)

    Applies frequency-domain amplitude perturbation on the VAE-encoded latent z ∈ R^{B×4×h×w}:
      1. Perform a 2D FFT on z, separating amplitude A(z) and phase P(z).
      2. Generate Gaussian noise δ ~ N(0, α²) and multiply it by the foreground-adaptive
         coefficient (1 + M_down).
      3. Perturb the amplitude: A'(z) = A(z) + δ.
      4. Reconstruct z' = IFFT(A'(z) · e^{jP(z)}) using A'(z) and the original phase P(z).

    The phase is preserved (maintaining the high-level semantic structure) while the amplitude
    is perturbed (broadening the semantic feature distribution), enabling the model to learn a
    wider latent-space distribution and improving generalization to novel samples.

    Complementary to PFA: PFA broadens the low-level texture distribution in pixel space, while
    LFA broadens the high-level semantic distribution in latent space. Together they form the
    DFDB (Dual-space Frequency Distribution Broadening) framework.

    Args:
        alpha: perturbation strength coefficient that controls the noise amplitude (default 0.1)
        prob:  perturbation trigger probability (default 0.3)
    """
    def __init__(self, alpha=0.1, prob=0.3):
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, z, mask=None):
        """
        Args:
            z:    VAE-encoded latent variable [B, C, h, w]  (typically C=4, h=w=64)
            mask: foreground mask [B, H, W] or [B, 1, H, W], used for adaptive weighting;
                  if None, uniform global perturbation is applied
        Returns:
            z_aug: perturbed latent variable, same shape as z
        """
        if random.random() > self.prob:
            return z

        z_freq = torch.fft.fft2(z)
        amplitude = torch.abs(z_freq)
        phase = torch.angle(z_freq)

        noise = torch.randn_like(amplitude) * self.alpha

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask_ds = F.interpolate(mask.float(), size=z.shape[-2:], mode='nearest')
            noise = noise * (1.0 + mask_ds)

        amplitude_aug = amplitude + noise

        z_aug = torch.polar(amplitude_aug, phase)
        z_aug = torch.fft.ifft2(z_aug).real

        z_mean = z.mean()
        z_std = z.std()
        z_aug = z_aug.clamp(z_mean - 4 * z_std, z_mean + 4 * z_std)

        return z_aug


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("=== PFA (DoublePerturb) test ===")
    pfa = DoublePerturb(64).to(device)
    img_q = torch.randn(1, 64, 32, 32).to(device)
    mask_q = torch.randn(1, 32, 32).to(device)
    spixel_mask_q = torch.randn(1, 64, 32, 32).to(device)
    y = pfa(img_q, mask_q, spixel_mask_q)
    print("PFA input:", img_q.shape, "-> output:", y.shape)

    print("\n=== LFA (LatentFrequencyPerturb) test ===")
    lfa = LatentFrequencyPerturb(alpha=0.1, prob=1.0).to(device)
    z = torch.randn(2, 4, 64, 64).to(device)
    mask_latent = torch.randint(0, 2, (2, 512, 512)).float().to(device)
    z_aug = lfa(z, mask=mask_latent)
    print("LFA input:", z.shape, "-> output:", z_aug.shape)
    print(f"LFA diff norm: {(z_aug - z).norm().item():.4f}")
