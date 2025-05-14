'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, perform_tilt
import fastmri

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device,):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)



def kspace2rss(kspace):    ####  这个可以
    """
    从多通道 k-space 数据计算 RSS 图像
    :param kspace: 输入 k-space 数据 (1, num_coils, H, W, 2) (包含实部/虚部)
    :return: RSS 图像 (H, W)
    """
    # Step 1: 计算 IFFT，转换到图像空间
    image_space = fastmri.ifft2c(kspace)  # (num_coils, H, W, 2)

    # Step 2: 计算幅度（去掉复数部分）
    abs_images = fastmri.complex_abs(image_space)  # (num_coils, H, W)

    # Step 3: 计算 RSS（多通道合成单通道）
    rss_image = fastmri.rss(abs_images, dim=1).unsqueeze(1) # (H, W)

    return rss_image


def rss_to_kspace(rss_image, csm):
    """
    将 RSS 图像转换回 k-space 数据
    :param rss_image: 单通道 RSS 图像 (H, W)
    :param csm: 线圈灵敏度图 (num_coils, H, W, 2)
    :return: k-space 数据 (num_coils, H, W, 2)
    """
    # Step 1: 确保 rss_image 形状匹配
    # rss_image = rss_image.unsqueeze(0)  # (1, H, W)
    rss_image = rss_image.unsqueeze(-1)  # (1, 1, H, W, 1) 适配复数格式

    # Step 2: 计算 Coil-wise 图像空间数据
    image_space = csm * rss_image  # (num_coils, H, W, 2)

    # Step 3: 计算 k-space (FFT)
    kspace = fastmri.fft2c(image_space)  # (num_coils, H, W, 2)
    
    return kspace

@register_operator(name='mri_acce')
class InpaintingOperator(LinearOperator):
    '''This operator get mri forward masked image.'''
    def __init__(self, device,):
        self.device = device
    
    def forward(self, x, mask, csm, img_min, img_max, **kwargs):
        """
        执行 mri_forward 处理 (支持归一化与反归一化)
        
        Args:
            x       : 输入 RSS 图像 (batch, 1, H, W)，归一化到 [-1, 1]
            mask    : 采样掩码 (1, 1, H, 1) 或 (batch, 1, H, W, 1)
            csm     : 线圈灵敏度图 (batch, num_coils, H, W, 2)
            img_min : 原始 RSS 图像的最小值（用于反归一化）
            img_max : 原始 RSS 图像的最大值（用于反归一化）

        Returns:
            再归一化到 [-1, 1] 范围内的 RSS 图像 (batch, 1, H, W)
        """
        # Step 0: 如果输入是 3 通道，取第 1 通道并记录
        will_expand_back = False
        if x.shape[1] == 3:
            x = x[:, 0:1, :, :]  # 取第一个通道
            will_expand_back = True
        
        # Step 1: 反归一化到原始强度
        x_phys = ((x + 1) / 2) * (img_max - img_min) + img_min  # ∈ [min, max]

        # Step 2: RSS 图像 → k-space（结合 CSM）
        k_reconstructed = rss_to_kspace(x_phys, csm)  # (B, Nc, H, W, 2)

        # Step 3: 数据一致性修正
        k_dc = mask * k_reconstructed
        # Step 4: 修正后的 k-space → RSS 图像
        rss_corrected = kspace2rss(k_dc)  # (B, 1, H, W)

        # Step 5: 再归一化到 [-1, 1]
        rss_normalized =((rss_corrected - img_min) / (img_max - img_min)) * 2 - 1
        # Step 6: 如果最开始是 3 通道，则复制回去
        if will_expand_back:
            rss_normalized = rss_normalized.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        return rss_normalized
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
    
class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='turbulence')
class TurbulenceOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, tilt, **kwargs):
        tilt_data = perform_tilt(data, tilt, image_size=data.shape[-1], device=data.device)
        blur_tilt_data = self.apply_kernel(tilt_data, kernel)
        return blur_tilt_data

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img
    
@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        for param in blur_model.parameters():
            param.requires_grad = False
        return blur_model
    
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred



    
    
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, noise_scale):
        self.sigma = noise_scale
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='impulse')
class ImpulseNoise(Noise):
    def __init__(self, noise_scale):
        self.p = noise_scale
    
    def forward(self, data):
        mask = torch.rand_like(data) < self.p
        replace = torch.where(torch.rand_like(data) < 0.5, 
                             torch.zeros_like(data), 
                             torch.ones_like(data))
        mask = mask.float()
        return data * (1 - mask) + replace * mask

@register_noise(name='shot')
class ShotNoise(Noise):
    def __init__(self, noise_scale):
        self.lam = noise_scale
    
    def forward(self, data):
        # 泊松分布采样需要非负输入，确保数据在有效范围
        scaled_data = data.clamp(0.0, 1.0) * self.lam
        noisy = torch.poisson(scaled_data) / self.lam
        return noisy.clamp(0.0, 1.0)

@register_noise(name='speckle')
class SpeckleNoise(Noise):
    def __init__(self, noise_scale):
        self.var = noise_scale
    
    def forward(self, data):
        epsilon = torch.randn_like(data) * (self.var ** 0.5)
        return data * (1 + epsilon)
    

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)