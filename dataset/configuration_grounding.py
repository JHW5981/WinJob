from types import SimpleNamespace
from typing import Dict, List, Any
from transformers import PretrainedConfig
import os
from typing import Union
                          
class DatasetConfig(SimpleNamespace):
    def __init__(self,
                 data_path={
                     "/home/zjr2022/datasets/train_vizwiz.json": 1500,
                 },
                 conv_template_name="phi_3",
                 image_aspect_ratio='anyres',
                 batch_size=5,
                 workers=2,
                 world_size=1,
                 gradient_accumulation_steps=1,
                 dataset_name=None,
                 use_bounding_box=True,
                 patch_area=196,
                 ):
        super().__init__(
            data_path=data_path,
            conv_template_name=conv_template_name,
            image_aspect_ratio=image_aspect_ratio,
            batch_size=batch_size,
            workers=workers,
            world_size=world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataset_name=dataset_name,
            use_bounding_box=use_bounding_box,
            patch_area=patch_area,
        )

class ImageProcessorConfig(PretrainedConfig):
    def __init__(
        self,
        do_resize: bool = True,
        grids: List[List] = [
                                [384, 384],
                                [384, 768],
                                [384, 1152],
                                [384, 1536],
                                [384, 1920],
                                [384, 2304],
                                [384, 2688],
                                [384, 3072],
                                [384, 3456],
                                [768, 384],
                                [768, 768],
                                [768, 1152],
                                [768, 1536],
                                [1152, 384],
                                [1152, 768],
                                [1152, 1152],
                                [1536, 384],
                                [1536, 768],
                                [1920, 384],
                                [2304, 384],
                                [2688, 384],
                                [3072, 384],
                                [3456, 384]
                            ],
        image_mean: List[float] = [0.5, 0.5, 0.5],
        image_std: List[float] = [0.5, 0.5, 0.5],
        interpolation_mode: str = "bicubic",
        resize_mode: str = "squash",
        size: List[int] = [384, 384],
        **kwargs,
    ):
        self.do_resize = do_resize
        self.grids = grids
        self.image_mean = image_mean
        self.image_std = image_std
        self.interpolation_mode = interpolation_mode
        self.resize_mode = resize_mode
        self.size = size

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict 
        config_dict = config_dict["image_processor_config"]

        return cls.from_dict(config_dict, **kwargs)

if __name__ == "__main__":
    dataset_config = DatasetConfig()
    print(dataset_config)

