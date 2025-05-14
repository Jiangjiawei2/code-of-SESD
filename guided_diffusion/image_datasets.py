import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import fastmri
import os
import h5py as h5



class Normalizer:
    def __init__(self, mode='image', eps=1e-8):
        self.mode = mode
        self.eps = eps
        self._cache = {}

    def normalize(self, img, tag=None):
        if self.mode == 'image':
            img = img.float()
            img_min = img.min()
            img_max = img.max()
            scale = img_max - img_min
            scale = scale if scale > self.eps else self.eps  # 防止除以0

            norm_img = ((img - img_min) / scale) * 2 - 1

            if tag is not None:
                self._cache[tag] = (img_min, img_max)

            return norm_img, img_min, img_max
        else:
            raise NotImplementedError("Only 'image' mode is implemented.")

    def denormalize(self, norm_img, img_min=None, img_max=None, tag=None):
        if self.mode == 'image':
            norm_img = norm_img.float()
            if tag is not None:
                if tag not in self._cache:
                    raise ValueError(f"No cached min/max found for tag: {tag}")
                img_min, img_max = self._cache[tag]

            if img_min is None or img_max is None:
                raise ValueError("Must provide img_min and img_max for denormalization.")

            scale = img_max - img_min
            scale = scale if scale > self.eps else self.eps  # 防止除以0

            return ((norm_img + 1) / 2) * scale + img_min
        else:
            raise NotImplementedError("Only 'image' mode is implemented.")
        
class m4raw_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size, class_cond=False, transforms=None):
        super().__init__()
        
        self.data_dir = root
        self.acceleration = 4
        self.modal_ref = "T1"
        self.modal_rec = "T2"
        self.image_size = image_size
        self.class_cond = class_cond
        
        # 自定义的变换，如果需要调整图像大小
        # self.transforms = transforms or transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     # 如果原始数据已经标准化到[-1,1]，则不需要额外的标准化
        # ])
        
        filenames = os.listdir(self.data_dir)
        self.filenames_rec = [f.split('.')[0] for f in filenames if self.modal_rec in f]
        self.filenames_ref = [f.split('.')[0] for f in filenames if self.modal_ref in f]

        self.pairs = []
        for f_ref in self.filenames_ref:
            f_rec = f_ref.replace(self.modal_ref, self.modal_rec)
            if f_rec in self.filenames_rec:
                self.pairs.append((f_ref, f_rec))
        
        self.normalizer = Normalizer()  # 您需要实现这个类

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename_ref, filename_rec = self.pairs[idx]
        with h5.File(os.path.join(self.data_dir, f'{filename_ref}.h5'), 'r') as f_ref, \
             h5.File(os.path.join(self.data_dir, f'{filename_rec}.h5'), 'r') as f_rec:
                 
            kspace_ref = f_ref['kspace'][()]
            kspace_rec = f_rec['kspace'][()]
            
            slice_idx = np.random.randint(0, kspace_rec.shape[0])
            k_rec = kspace_rec[slice_idx]
            k_ref = kspace_ref[slice_idx]

            csm = self.estimate_csm_from_kspace(torch.from_numpy(k_rec))
            mask = self.generate_random_mask(k_rec.shape[-1], acceleration=self.acceleration)
            k_under = self.apply_mask(torch.from_numpy(k_rec), mask)
            
            rec_rss = self.kspace2rss(torch.view_as_real(torch.from_numpy(k_rec))).cpu().numpy()
            under_img = self.kspace2rss(k_under).cpu().numpy()
            
            rec_rss_tensor = torch.from_numpy(rec_rss).float().unsqueeze(0)  # [1, H, W]
            under_img_tensor = torch.from_numpy(under_img).float().unsqueeze(0)
            
            rec_rss_norm, img_min, img_max = self.normalizer.normalize(rec_rss_tensor, tag=filename_rec)
            under_img_norm, _, _ = self.normalizer.normalize(under_img_tensor)
            
           # 核心修复：确保返回格式是 (image_data, condition_dict)
            image_data = rec_rss_norm  # 已经是[-1, 1]范围的归一化数据
            
            # 确保图像数据形状正确 [channels, height, width]
            if image_data.shape[0] == 1:  # 如果是单通道
                # 在训练前进行尺寸调整
                if image_data.shape[1] != self.image_size or image_data.shape[2] != self.image_size:
                    # 可以使用interpolate进行调整
                    image_data = torch.nn.functional.interpolate(
                        image_data.unsqueeze(0),  # [1, 1, H, W]
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # 回到 [1, image_size, image_size]
                    
                image_data = image_data.repeat(3, 1, 1)  # 将通道从[1, H, W]变成[3, H, W]
                
            # 条件字典
            cond_dict = {}
            if self.class_cond:
                # 如果需要类条件，可以在这里设置
                # 例如，使用不同的MRI模态作为条件
                cond_dict["y"] = np.array([1], dtype=np.int64)  # 假设1表示T2
            
            return image_data, cond_dict
            
    def estimate_csm_from_kspace(self, kspace):
        """
        估计 CSM（线圈灵敏度图）
        :param kspace: k-space 数据 (coils, H, W, 2)
        :return: 线圈灵敏度图 (coils, H, W, 2)
        """
        if kspace.ndim == 3 and torch.is_complex(kspace):
            # [Nc, H, W] complex → [Nc, H, W, 2]
            kspace = torch.view_as_real(kspace)
        elif kspace.ndim != 4 or kspace.shape[-1] != 2:
            raise ValueError(f"Expected k-space shape [Nc, H, W, 2] or complex [Nc, H, W], but got {kspace.shape}")
        img_space = fastmri.ifft2c(kspace)  # (coils, H, W, 2)

        # 计算 RSS（多通道合成单通道）
        rss_image = fastmri.rss(fastmri.complex_abs(img_space), dim=0)  # (1, 1, H, W)

        # 归一化计算 CSM，扩展维度确保广播正确
        csm = img_space / (rss_image.squeeze(0).squeeze(0).unsqueeze(-1) + 1e-8)

        return csm  # (coils, H, W, 2)  

    
    def kspace2rss(self, kspace):    ####  这个可以
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
        rss_image = fastmri.rss(abs_images, dim=0) # (H, W)

        return rss_image


    
    def rss_to_kspace(self, rss_image, csm):
        """
        将 RSS 图像转换回 k-space 数据
        :param rss_image: 单通道 RSS 图像 (H, W)
        :param csm: 线圈灵敏度图 (num_coils, H, W, 2)
        :return: k-space 数据 (num_coils, H, W, 2)
        """
        # Step 1: 确保 rss_image 形状匹配
        # rss_image = rss_image.unsqueeze(0)  # (1, H, W)
        rss_image = rss_image.unsqueeze(-1)  # (1, H, W, 1) 适配复数格式

        # Step 2: 计算 Coil-wise 图像空间数据
        image_space = csm * rss_image  # (num_coils, H, W, 2)

        # Step 3: 计算 k-space (FFT)
        kspace = fastmri.fft2c(image_space)  # (num_coils, H, W, 2)
        
        return kspace

    def generate_random_mask(self, width, acceleration=4, center_fraction=0.08):
        num_low_freqs = int(width * center_fraction)
        mask = torch.zeros(width)
        center_start = (width - num_low_freqs) // 2
        mask[center_start:center_start + num_low_freqs] = 1

        prob = 1.0 - mask
        num_to_select = int((width - num_low_freqs) / acceleration)
        rand_inds = torch.multinomial(prob, num_to_select, replacement=False)
        mask[rand_inds] = 1

        return mask.view(1, width, 1)

    def apply_mask(self, kspace, mask):
        if kspace.ndim == 4:
            kspace = torch.view_as_complex(kspace)
        masked = kspace * mask.squeeze()
        return torch.view_as_real(masked)
    
    
def load_mri_data(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_flip=True,
):
    """
    加载MRI数据集用于扩散模型训练
    
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        image_size: 图像大小
        class_cond: 是否使用类别条件
        deterministic: 是否使用确定性数据加载
        random_flip: 是否随机翻转（MRI可能不需要这个）
    
    Returns:
        一个可迭代对象，提供批次数据
    """
    if not data_dir:
        raise ValueError("未指定数据目录")
    
    # 创建您的MRI数据集
    dataset = m4raw_Dataset(
        root=data_dir,
        image_size=image_size,
        class_cond=class_cond,
    )
    
    # 创建数据加载器
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    # 包装为无限迭代器
    while True:
        yield from loader
        
def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_flip=True,
):
    """
    替换原始的load_data函数，使用MRI数据加载器
    """
    return load_mri_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=deterministic,
        random_flip=random_flip,
    )
       
        
        
# def load_data(
#     *,
#     data_dir,
#     batch_size,
#     image_size,
#     class_cond=False,
#     deterministic=False,
#     random_crop=False,
#     random_flip=True,
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.

#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.

#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     :param random_crop: if True, randomly crop the images for augmentation.
#     :param random_flip: if True, randomly flip the images for augmentation.
#     """
#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#         random_crop=random_crop,
#         random_flip=random_flip,
#     )
#     if deterministic:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
#         )
#     while True:
#         yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
