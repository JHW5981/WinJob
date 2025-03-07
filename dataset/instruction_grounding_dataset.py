from configuration_grounding import DatasetConfig
from transformers.utils import PaddingStrategy
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict, Optional
from dataclasses import dataclass
from PIL import ImageDraw
from typing import Dict
from PIL import Image
import numpy as np
import random
import json
import torch

class AceRead2Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 dataset_config: DatasetConfig,
                 processor,
                 ):
        super(AceRead2Dataset, self).__init__()
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

        self.processor = processor
        self.list_data_dict = list_data_dict
        random.shuffle(self.list_data_dict)

    def __len__(self):
        return len(self.list_data_dict)

    def check_coordinate(self, c):
        if c[0] < 0:
            c[0] = 0
        if c[1] < 0:
            c[1] = 0
        return c

    def process_coordinate(self, coordinate, origin_h, origin_w, resized_h, resized_w):
        h_ratio = resized_h/origin_h
        w_ratio = resized_w/origin_w
        new_coordinate = []
        for c in coordinate:
            c = self.check_coordinate(c)
            new_c = [c[0]*h_ratio if c[0]*h_ratio < resized_h else resized_h, c[1]*w_ratio if c[1]*w_ratio < resized_w else resized_w]
            new_coordinate.append(new_c)
        return new_coordinate

    def __getitem__(self, i):
        source = self.list_data_dict[i]
        image_file = source["image"][0]
        conversation = source["conversations"]
        answer_grounding = [[item['y'], item['x']] for item in source["answer_grounding"]]

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {
                    "type": "text",
                    "text": conversation[0]['value']
                }
            ],
        }]
        output_content = conversation[1]['value']
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() for key, value in inputs.items()}
        instruction = inputs
        
        response = self.processor.tokenizer(f"{output_content}", add_special_tokens=False)
        input_ids = (
                    instruction["input_ids"][0] + response["input_ids"] + [self.processor.tokenizer.pad_token_id]
            )
        attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
                    [-100] * len(instruction["input_ids"][0])
                    + response["input_ids"]
                    + [self.processor.tokenizer.pad_token_id]
            )

        # process image
        img = Image.open(image_file)
        channel = 3
        origin_h = img.height
        origin_w = img.width
        image_grid_thw = inputs['image_grid_thw']
        grid_h = image_grid_thw[0][1] 
        grid_w = image_grid_thw[0][2]  
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        resized_h = grid_h*patch_size
        resized_w = grid_w*patch_size
        temporal_patch_size = self.processor.image_processor.temporal_patch_size
        pixel_values = np.array(inputs['pixel_values'])

        # get original image
        tokens_grid = pixel_values.reshape(1, grid_h//merge_size,\
                                            grid_w//merge_size, \
                                            merge_size, merge_size,\
                                            channel, temporal_patch_size,\
                                            patch_size, patch_size)

        tokens_grid = tokens_grid.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
        tokens_grid = tokens_grid.reshape(temporal_patch_size, channel,\
                                        grid_h*patch_size,\
                                        grid_w*patch_size)
        tokens_grid = tokens_grid[0]
        tokens_grid = tokens_grid.transpose(1, 2, 0) # (xxx, xxx, 3)
        original_pixel = (tokens_grid - tokens_grid.min()) / (tokens_grid.max() - tokens_grid.min())
        
        # process coordinate
        answer_grounding = self.process_coordinate(answer_grounding, origin_h, origin_w, resized_h, resized_w)
        
        # get possibilites
        mask_img = Image.new('L', (resized_w, resized_h), 0)
        draw = ImageDraw.Draw(mask_img)
        polygon_points = [(x, y) for (y, x) in answer_grounding]
        draw.polygon(polygon_points, outline=1, fill=1)
        mask_np = np.array(mask_img, dtype=np.uint8)
        mask_np = np.expand_dims(mask_np, axis=0)
        mask_np = np.expand_dims(mask_np, axis=0)
        mask_np = np.repeat(mask_np, channel, axis=1)

        # copy from Qwen2VLImageProcessor._preprocess
        if mask_np.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(mask_np[-1][np.newaxis], temporal_patch_size - 1, axis=0)
            mask_np = np.concatenate([mask_np, repeats], axis=0)
        grid_t = mask_np.shape[0] // temporal_patch_size
        patches = mask_np.reshape(
                    grid_t,
                    temporal_patch_size,
                    channel,
                    grid_h // merge_size,
                    merge_size,
                    patch_size,
                    grid_w // merge_size,
                    merge_size,
                    patch_size,
                )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
                    grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
                )
        possibilites = flatten_patches.sum(axis=1) / flatten_patches.sum()
        possibilites_recover = possibilites.reshape(1, grid_h//merge_size,\
                                            grid_w//merge_size, \
                                            merge_size, merge_size, 1, 1, 1, 1)
        possibilites_recover = possibilites_recover.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
        possibilites_recover = possibilites_recover.reshape(1, 1,\
                                        grid_h*1,\
                                        grid_w*1).squeeze()

        data_dict = {
            "image": original_pixel,
            "conversation": text + f" {output_content}" + "<|endoftext|>",
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "pixel_values": torch.tensor(pixel_values),
            "answer_grounding": answer_grounding,
            "possibilites": torch.tensor(possibilites),
            "possibilites_recover": possibilites_recover,
        }
    
        return data_dict

# copied from transformers/data/data_collator.py
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorForAceRead2Dataset:

    processor: object = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features):
        label_name = "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        input_ids_and_attention_mask = [{k: v for k, v in feature.items() if k in ["input_ids", "attention_mask"]} for feature in features]
        images = [feature["image"] for feature in features]
        conversation = [feature["conversation"] for feature in features]
        pixel_values = [feature["pixel_values"] for feature in features]
        answer_grounding = [feature["answer_grounding"] for feature in features]
        possibilites = [feature["possibilites"] for feature in features]
        possibilites_recover = [feature["possibilites_recover"] for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.processor.tokenizer,
            input_ids_and_attention_mask,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.processor.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if self.return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif self.return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None
        batch['image'] = images
        batch['conversation'] = conversation
        batch['pixel_values'] = pixel_values
        batch['answer_grounding'] = answer_grounding
        batch['possibilites'] = possibilites
        batch['possibilites_recover'] = possibilites_recover
        return batch

def make_instruction_grounding_dataset(
    dataset_config,
    processor,
):
    train_dataset = AceRead2Dataset(
        dataset_config=dataset_config,
        processor=processor
    )
    data_collator = DataCollatorForAceRead2Dataset(
        processor=processor,
        padding=True
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    return data_loader, len(train_dataset)

if __name__ == "__main__":
    from transformers import Qwen2_5_VLProcessor
    dataset_config = DatasetConfig()
    processor = Qwen2_5_VLProcessor.from_pretrained("/home/jihuawei2/projects/WinJob/pretrained_weight/Qwen2.5-VL-3B-Instruct")
    dataloader, dataset_size = make_instruction_grounding_dataset(dataset_config, processor)
    for data in dataloader:
        print(data.keys())
        # print(data['image'].shape)
        # print(data['answer_grounding'])
        # print(data['conversation'])
        # print(data['possibilites_recover'])
        print(data['input_ids'].shape)
        print(data['attention_mask'].shape)
        break
    
    
    
    
    
    
    
    # dataset = AceRead2Dataset(dataset_config, processor)
    # inputs = dataset[0]
    # image = inputs['image']
    # answer_grounding = inputs['answer_grounding']
    # conversation = inputs['conversation']
    # possibilites_recover = inputs['possibilites_recover']
    # mask_uint8 = ((possibilites_recover>0) * 255).astype(np.uint8)
    # img = Image.fromarray(mask_uint8, mode='L')
    # img.save('mask_image.png')

    # print(image.shape)
    # print(answer_grounding)
    # print(conversation)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from matplotlib.patches import Polygon

    # # 如果 image 是 torch.Tensor，则转换为 numpy 数组
    # if isinstance(image, torch.Tensor):
    #     image_np = image.cpu().numpy()
    # else:
    #     image_np = image

    # # 确保图像数值在 [0, 1] 范围内
    # image_np = np.clip(image_np, 0, 1)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_np)

    # ax = plt.gca()

    # # 如果 answer_grounding 中的点数大于等于 3，则构成多边形
    # if len(answer_grounding) >= 3:
    #     # answer_grounding 点格式为 [y, x]，转换为 [x, y]
    #     polygon_points = [[pt[1], pt[0]] for pt in answer_grounding]
    #     # 创建一个多边形补丁，closed=True 表示闭合多边形，fill=False 表示不填充颜色
    #     poly_patch = Polygon(polygon_points, closed=True, fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(poly_patch)
    # else:
    #     # 否则仅绘制散点
    #     for pt in answer_grounding:
    #         y, x = pt
    #         plt.scatter(x, y, s=100, c='red', marker='o')

    # plt.title("Image with Answer Grounding Polygon")
    # plt.axis('off')

    # # 保存图片到文件，保存前调用 plt.savefig，再调用 plt.show()
    # plt.savefig('output_polygon.png', bbox_inches='tight')
    # plt.show()
