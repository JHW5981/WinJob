# -*- encoding: utf-8 -*-
'''
@File    :   instruction_grounding_dataset.py
@Time    :   2025/02/22 16:42:00
@Author  :   zjr2022
'''

import sys
sys.path.append('/home/jihuawei2/projects/WinJob')
sys.path.append('/home/jihuawei2/projects/WinJob/dataset')

from dataset.configuration_grounding import DatasetConfig
from torch.utils.data import Dataset
from dataset.data_utils import DataInfo
from dataset.utils import (
    process_anyres_image_dataset,
    get_modality_length_grouped_indices,
    get_length_grouped_indices,
    preprocess,
    IGNORE_INDEX,
)
from torch.utils.data import DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Sequence
from dataclasses import dataclass
from glob import glob
from PIL import Image
import transformers
import numpy as np
import random
import copy
import json
import torch
import cv2
import os

DEFAULT_IMAGE_TOKEN = "<image>"

# The following code is reused from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/train/sft_data_utils.py
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 dataset_config: DatasetConfig,
                 tokenizer1: transformers.PreTrainedTokenizer, # Phi-3 tokenizer
                 tokenizer2: transformers.PreTrainedTokenizer, # Siglip tokenizer
                 image_processor,
                 ):
        super(LazySupervisedDataset, self).__init__()
        self.dataset_config = dataset_config
        if isinstance(self.dataset_config.data_path, Dict):
            list_data_dict = []
            for json_file, n_sample in self.dataset_config.data_path.items():
                d_json = json.load(open(json_file, "r"))
                if n_sample > len(d_json):
                    list_data_dict.extend(random.Random(42).choices(d_json, k=n_sample))
                else:
                    list_data_dict.extend(random.Random(42).sample(d_json, k=n_sample))
        else:
            raise ValueError(f"Only Dict is acceptable, you give unknown data_path type: {type(self.dataset_config.data_path)}")

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer1 = tokenizer1 # Phi-3 tokenizer
        self.tokenizer2 = tokenizer2 # Siglip tokenizer
        self.image_processor = image_processor
        self.conv_template_name = self.dataset_config.conv_template_name
        self.list_data_dict = list_data_dict
        random.shuffle(self.list_data_dict)

        self.anyres_grids = image_processor.grids

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(str(conv['value']).split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _process_single_image(self, image_file) -> Dict[str, torch.Tensor]:
        image_file_fullpath = image_file
        success = True
        try:
            image = Image.open(image_file_fullpath).convert('RGB')
        except:
            print(f"error opening the file: {image_file_fullpath}")
            success = False
            return success, None, None
        processor = self.image_processor
        img_size = image.size
        if self.dataset_config.image_aspect_ratio == "anyres":
            # Return image shape: [N_patch, C, H, W]
            image = process_anyres_image_dataset(image, processor, self.anyres_grids, processor.size)
        else:
            raise ValueError(f"image_aspect_ratio must be anyres, but got {self.dataset_config.image_aspect_ratio}")
        
        return success, image, img_size
    
    def _check_img_token_nums(self, source):
        keep_sample = True
        if 'image' not in source:
            # Make sure no <image> token in text-only samples.
            for conv in source["conversations"]:
                n_img_token = conv["value"].count(DEFAULT_IMAGE_TOKEN)
                if n_img_token > 0:
                    keep_sample = False
                    break
            return keep_sample, source
        n_image = len(source['image']) if isinstance(source['image'], list) else 1
        if n_image > 1:
            # FIXME: the checker below doesn't work for mantis. Currently only check for single image data.
            return keep_sample, source
        for conv in source["conversations"]:
            if conv["from"] == "human":
                n_img_token = conv["value"].count(DEFAULT_IMAGE_TOKEN)
                if not n_img_token == n_image:
                    # print(source)
                    conv["value"] = conv["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    conv["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" * n_image + conv["value"]
                break
        return keep_sample, source

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        keep_sample, sources = self._check_img_token_nums(sources)
        if not keep_sample:
            return self.__getitem__(i+1)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # Add the system prompt.
        system_round = {
              "from": "system",
              "value": "A chat between a curious user and an artificial intelligence assistant. The assistant gives concise answers to the user's questions."
            }
        if sources[0]["conversations"][0]["from"] != "system":
            sources[0]["conversations"] = [system_round] + sources[0]["conversations"]

        if 'image' in sources[0]:
            has_image = True
            image_file = sources[0]['image']
            assert isinstance(image_file, list)
            # FIXME: Skipping samples with more than 4 images to avoid OOM issue.
            if len(image_file) > 4:
                return self.__getitem__(i+1)
            image = []
            img_size = []
            img_patch_num = [] # added by JHW5981 FIXME: multi images will cause error, fix in the future
            for single_image in image_file:
                success, image_i, img_size_i = self._process_single_image(single_image)
                if not success:
                    # Skip the entire sample if one of the images can't be opened.
                    return self.__getitem__(i+1)
                image.append(image_i)
                img_size.append(img_size_i)
                img_patch_num.append(image_i.shape[0])
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            has_image = False
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer1,
            conv_template_name=self.conv_template_name)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            
        # add semantic tokens by JHW5981
        query = sources[0][1]['content'].replace(f"<image>\n", "") # FIXME: not using fixed <image> but attribute from tokenizer1, considering more than one query sentence
        semantic_inputs = self.tokenizer2(query, padding="max_length", return_tensors="pt")
        data_dict['lang_y'] = semantic_inputs

        # image exist in the data
        if has_image:
            assert isinstance(image, list)
            # Multi-image, each image can be of 4-dim (anyres) or 3-dim (base res)
            data_dict['vision_x'] = image 
            data_dict['image_size'] = img_size
            data_dict['image_patch_num'] = img_patch_num
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.transforms[0].size # FIXME: Hardcoded workaround to work with torchvision.Compose()
            data_dict['vision_x'] = torch.zeros(1, 1, 3, crop_size[0], crop_size[1]) # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = crop_size
            data_dict['image_patch_num'] = img_patch_num
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # added by JHW5981
        lang_y = torch.cat([i['lang_y']['input_ids'] for i in instances], dim=0)

        batch = dict(
            lang_x=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            lang_y={'input_ids': lang_y},
        )

        if 'vision_x' in instances[0]:
            images = [instance['vision_x'] for instance in instances]
            image_size = [instance['image_size'] for instance in instances]
            image_patch_num = [instance['image_patch_num'] for instance in instances]
            batch['image_size'] = image_size
            batch['image_patch_num'] = image_patch_num
            batch['vision_x'] = images
            batch['image_size'] = image_size

        return batch

class InstructionGroundingDataset(LazySupervisedDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer1: transformers.PreTrainedTokenizer,
        tokenizer2: transformers.PreTrainedTokenizer,
        image_processor,
    ):
        super().__init__(dataset_config, tokenizer1, tokenizer2, image_processor)
        # patch_area 是网格的面积，如 196
        self.patch_area = dataset_config.patch_area
        # patch_size 是网格边长，如 14
        self.patch_size = int(self.patch_area**0.5)

    def _process_single_image(self, image_file) -> Dict[str, torch.Tensor]:
        success, image, img_size = super()._process_single_image(image_file)
        return success, image, img_size

    def _process_answer_grounding(self, data: Dict, image_size: Tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        patch_size = self.patch_size
        
        # 计算 patch 数量
        patch_area_h = height // patch_size
        patch_area_w = width // patch_size
       # print(data)
        total_patches = patch_area_h * patch_area_w
        position_distribution = np.zeros(total_patches)
        total_pixels = height * width

        # 构建答案区域的掩码
        answer_mask = np.zeros((height, width), dtype=np.uint8)
        grounding_coords = data["answer_grounding"]
        points = [(int(coord["x"]), int(coord["y"])) for coord in grounding_coords]
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(answer_mask, [points], 1)

        # 遍历每个 patch
        patch_index = 0
        for i in range(patch_area_h):
            for j in range(patch_area_w):
                patch_y1 = i * patch_size
                patch_y2 = patch_y1 + patch_size
                patch_x1 = j * patch_size
                patch_x2 = patch_x1 + patch_size
                patch = answer_mask[patch_y1:patch_y2, patch_x1:patch_x2]
                patch_pixels = patch_size * patch_size
                pixels_in_answer = np.sum(patch)
                if pixels_in_answer == 0:
                    position_distribution[patch_index] = 0
                elif pixels_in_answer == patch_pixels:
                    position_distribution[patch_index] = patch_pixels / total_pixels
                    
                else:
                    position_distribution[patch_index] = pixels_in_answer / total_pixels
                    #print(pixels_in_answer / total_pixels)
                patch_index += 1

        # 归一化，确保所有概率和为 1
        total_sum = np.sum(position_distribution)
        if total_sum != 0:
            position_distribution /= total_sum
            return torch.from_numpy(position_distribution).float()
        else:
            raise ValueError("Total sum is zero, indicating no valid regions for distribution.") 

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            data_dict = super().__getitem__(i)
            source = self.list_data_dict[i]
            image_size = data_dict['image_size'][0]  # Assuming single image for now
            position_distribution = self._process_answer_grounding(source, image_size)
            data_dict['position_distribution'] = position_distribution
            return data_dict
        except ValueError as e:
            return self.__getitem__(i+1)

    @property
    def lengths(self):
        return super().lengths

    @property
    def modality_lengths(self):
        return super().modality_lengths

class DataCollatorForGroundingDataset(DataCollatorForSupervisedDataset):
    def __init__(self, tokenizer, image_aspect_ratio):
        super().__init__(tokenizer, image_aspect_ratio)
        self.patch_area = int(14**2)  # Assuming 14x14 grid, adjust if necessary

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        
        position_distributions = []
        for instance in instances:
            if 'position_distribution' in instance:
                position_distributions.append(instance['position_distribution'])
            else:
                # If position_distribution is missing, create a uniform distribution
                position_distributions.append(torch.ones(self.patch_area) / self.patch_area)

        # 找到最大长度
        max_length = max([len(t) for t in position_distributions])
        padded_tensors = []
        for tensor in position_distributions:
            if len(tensor) < max_length:
                # 填充操作
                padding = torch.zeros(max_length - len(tensor))
                padded_tensor = torch.cat([tensor, padding])
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)

        batch['position_distributions'] = torch.stack(padded_tensors)

        return batch

class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

def make_instruction_grounding_dataset(
    dataset_config: DatasetConfig,
    tokenizer1: transformers.PreTrainedTokenizer,
    tokenizer2: transformers.PreTrainedTokenizer,
    image_processor,
):
    train_dataset = InstructionGroundingDataset(
        dataset_config=dataset_config,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        image_processor=image_processor
    )
    data_collator = DataCollatorForGroundingDataset(
        tokenizer=tokenizer1,
        image_aspect_ratio=dataset_config.image_aspect_ratio
    )

    # Use length grouped sampler for more balanced GPU usages.
    lengths = train_dataset.modality_lengths
    sampler = LengthGroupedSampler(
        dataset_config.batch_size,
        world_size=dataset_config.world_size * dataset_config.gradient_accumulation_steps,
        lengths=lengths,
        group_by_modality=True,
        generator=torch.Generator().manual_seed(42),
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=data_collator,
    )

    return DataInfo(
        name='instruction-grounding-mix',
        dataloader=data_loader,
        batch_size=dataset_config.batch_size,
        loss_multiplier=1.0,
        shared_epoch=None,
        sampler=sampler,
    ), len(train_dataset)

if __name__ == "__main__":
    from dataset.configuration_grounding import DatasetConfig, ImageProcessorConfig
    from dataset.image_processor import ImageProcessor
    from transformers import AutoTokenizer

    dataset_config = DatasetConfig()
    tokenizer1 = AutoTokenizer.from_pretrained("/home/jihuawei2/projects/AceRead/pretrain_weights/xgen-mm-phi3-mini-Instruct/xgen")
    tokenizer2 = AutoTokenizer.from_pretrained("/home/jihuawei2/projects/AceRead/pretrain_weights/xgen-mm-phi3-mini-Instruct/siglip-so400m-patch14-384")
    
    image_processor_config = ImageProcessorConfig()
    image_processor = ImageProcessor(image_processor_config)
    instruction_grounding_dataset, num_samples = make_instruction_grounding_dataset(
        dataset_config=dataset_config,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        image_processor=image_processor,
    )
    print("num_samples:", num_samples)
    print("anyres_grids:", instruction_grounding_dataset.dataloader.dataset.anyres_grids)
    for batch in instruction_grounding_dataset.dataloader:
        print(batch.keys())
        position_distributions = batch['position_distributions']
        vision_x = batch['vision_x']
        print("position_distributions shape:", position_distributions.shape)
        print("vision_x shape:", vision_x[1][0].shape)
        print("vision_x shape:", vision_x[2][0].shape)
        print("vision_x shape:", vision_x[3][0].shape)
        print("vision_x shape:", vision_x[4][0].shape)
        # if 'position_distributions' in batch:
        #     print("position_distributions shape:", batch['position_distributions'].shape)
        # if 'image_patch_num' in batch:
        #     print("image_patch_num:", batch['image_patch_num'])
        break
