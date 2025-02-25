from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from dataset.configuration_grounding import ImageProcessorConfig
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
import torchvision.transforms.functional as F
from transformers.image_utils import ImageInput
from typing import Union, Optional
from transformers.utils import TensorType
import torch
import random
from dataset.utils import (
    _setup_size,
    center_crop_or_pad,
)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    

class ResizeKeepRatio:
    """ Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation=InterpolationMode.BICUBIC,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string

class ImageProcessor(BaseImageProcessor): 
    def __init__(
        self,
        config: ImageProcessorConfig,
        **kwargs,
    ) -> None:
        self.do_resize = config.do_resize
        self.resize_mode = config.resize_mode
        self.interpolation_mode = config.interpolation_mode
        self.size = config.size if config.size is not None else (384, 384)
        self.grids = config.grids if config.grids is not None else [[384, 768],[768, 384],[768, 768],[1152, 384],[384,1152]]

        self.image_mean = config.image_mean if config.image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = config.image_std if config.image_std is not None else [0.5, 0.5, 0.5]
        super().__init__(**kwargs) 
    
    @classmethod
    def resize(cls, image_size, resize_mode, interpolation='bicubic', fill_color=0):
        interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC
        if resize_mode == 'longest':
            transforms = [
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=fill_color)
            ]
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transforms = [
                    Resize(image_size[0], interpolation=interpolation_mode)
                ]
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms = [ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]
        return transforms
    
    @classmethod
    def convert_rgb(cls, image):
        return image.convert("RGB")
            
    def _preprocess(self, 
                   images: ImageInput
                   ) -> torch.Tensor:
        transforms = self.resize(self.size,  self.resize_mode, self.interpolation_mode)
        transforms.extend([
            self.convert_rgb,
            ToTensor(),
            Normalize(mean=self.image_mean, std=self.image_std)
        ])
        composed_transforms = Compose(transforms)
        images_tensor = composed_transforms(images)
        return images_tensor           

    def process_anyres_image_model(self, image, processor_model, grid_pinpoints, processor_size): # 在单独处理图像时，需要提供processor_size，因为这个时候process_anyres_image传入的processor，image_processor的一个函数_preprocess
        image_patch = processor_model(image)# [processor(image_patch) for image_patch in image_patches]
        return image_patch

    def preprocess(self, 
                   images: ImageInput, 
                   return_tensors: Optional[Union[str, TensorType]] = None,
                   **kwargs) -> BatchFeature:
        image_aspect_ratio = 'anyres'
        new_images = []

        for image in images:
            image = self.process_anyres_image_model(image, self._preprocess, self.grids, self.size)
            new_images.append(image)
        
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        if image_aspect_ratio == 'anyres':
            new_images = BatchFeature(data={"pixel_values": new_images}, tensor_type=return_tensors)
        else:
            new_images = BatchFeature(data={"pixel_values": new_images.unsqueeze(1).unsqueeze(0)}, tensor_type=return_tensors)
        if len(new_images['pixel_values'].shape) == 3:
            new_images['pixel_values'] = new_images['pixel_values'].unsqueeze(0)
        return new_images