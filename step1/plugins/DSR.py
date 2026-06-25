# Dual-space Frequency Distribution Broadening (DFDB)
# ======================================================
# 双空间频率分布扩展框架，通过像素空间与潜空间的频域幅度扰动，
# 扩展小样本声呐图像训练中的特征分布，提升 diffusion 模型的鲁棒性与泛化能力。
#
# 包含两个互补模块：
#   PFA (Pixel-level Frequency Augmentation)  — 像素级频率增强（原 DoublePerturb）
#   LFA (Latent-level Frequency Augmentation) — 潜空间频率增强（VAE 编码后）

import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class RandomConv(nn.Module):
    """
    随机卷积模块：生成随机卷积核，对特征进行全局扰动
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
    掩码膨胀模块：扩大前景掩码范围，确保前景区域的完整性
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
# PFA: Pixel-level Frequency Augmentation (原 Dual Style Randomization)
# =====================================================================
class DoublePerturb(nn.Module):
    """
    像素级频率增强 (PFA / Pixel-level Frequency Augmentation)

    在 VAE 编码前对 RGB 像素图像进行频域幅度扰动：
      - 前景路径：从随机选取的超像素区域借用幅度，与前景区域的幅度混合
      - 全局路径：用随机卷积生成替代幅度，与原幅度融合
    保留相位（结构不变），只扰动幅度（扩展低层纹理分布），
    从而在小样本条件下丰富像素级特征多样性，降低过拟合风险。
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
    潜空间频率增强 (LFA / Latent-level Frequency Augmentation)

    在 VAE 编码后的潜空间 z ∈ R^{B×4×h×w} 上进行频域幅度扰动：
      1. 对 z 做 2D FFT，分离幅度 A(z) 和相位 P(z)
      2. 生成高斯噪声 δ ~ N(0, α²)，乘以前景自适应系数 (1 + M_down)
      3. 扰动幅度 A'(z) = A(z) + δ
      4. 用 A'(z) 与原相位 P(z) 重建 z' = IFFT(A'(z) · e^{jP(z)})

    保留相位（保持高层语义结构），扰动幅度（扩展语义特征分布），
    使模型学到更广泛的潜空间分布，增强对新样本的泛化能力。

    与 PFA 互补：PFA 在像素空间扩展低层纹理分布，LFA 在潜空间扩展高层语义分布，
    两者组合构成 DFDB (Dual-space Frequency Distribution Broadening) 框架。

    Args:
        alpha: 扰动强度系数，控制噪声幅度 (默认 0.1)
        prob:  扰动触发概率 (默认 0.3)
    """
    def __init__(self, alpha=0.1, prob=0.3):
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, z, mask=None):
        """
        Args:
            z:    VAE 编码后的潜变量 [B, C, h, w]  (通常 C=4, h=w=64)
            mask: 前景掩码 [B, H, W] 或 [B, 1, H, W]，用于自适应加权
                  若为 None 则全局均匀扰动
        Returns:
            z_aug: 扰动后的潜变量，形状同 z
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
