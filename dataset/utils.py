# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/12/26 22:30:46
@Author  :   JHW5981 
'''

import ast
import torch
import math
import numpy as np
from PIL import Image
from typing import Dict, Optional, Sequence, List
import transformers
import dataset.conversation as conversation_lib
import numbers
import torchvision.transforms.functional as F

IGNORE_INDEX = -100

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

from torchvision.transforms.functional import to_tensor

def process_anyres_image_dataset(image, processor_dataset, grid_pinpoints, processor_size=(384, 384)):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # FIXME: determine grid_pinpoints from image sizes.
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor_size[0])

    if len(patches) == 1:
        image_patches = patches
    else:
        image_original_resize = image.resize((processor_size[0], processor_size[0]))
        image_patches = [image_original_resize] + patches
    
    image_patches = processor_dataset(image_patches)['pixel_values'] # [processor(image_patch) for image_patch in image_patches]
    return image_patches

# def process_anyres_image(image, processor, grid_pinpoints):
#     """
#     Process an image with variable resolutions.

#     Args:
#         image (PIL.Image.Image): The input image to be processed.
#         processor: The image processor object.
#         grid_pinpoints (str): A string representation of a list of possible resolutions.

#     Returns:
#         torch.Tensor: A tensor containing the processed image patches.
#     """
#     # 解析可能的分辨率
#     if isinstance(grid_pinpoints, list):
#         possible_resolutions = grid_pinpoints
#     else:
#         possible_resolutions = ast.literal_eval(grid_pinpoints)
    
#     # 选择最佳分辨率并调整图像
#     best_resolution = select_best_resolution(image.size, possible_resolutions)
#     image_padded = resize_and_pad_image(image, best_resolution)

#     processor_size = processor.size
#     patches = divide_to_patches(image_padded, processor_size[0])

#     # 原始图像调整为目标大小
#     image_original_resize = image.resize((processor_size[0], processor_size[0]))
#     image_patches = [image_original_resize] + patches
#     own_processor=processor
#     processed_patches = []
#     for idx, image_patch in enumerate(image_patches):
#         processed_patch = own_processor(image_patch)

#         # 检查返回类型
#         if isinstance(processed_patch, torch.Tensor):
#             pixel_values = processed_patch
#         elif isinstance(processed_patch, list) and "pixel_values" in processed_patch:
#             # 先将列表转换为 numpy 数组，再转换为 torch.Tensor
#             pixel_values = torch.tensor(np.array(processed_patch["pixel_values"]))
#         elif isinstance(processed_patch, transformers.image_processing_base.BatchFeature):
#             if "pixel_values" in processed_patch:
#                 # 将 BatchFeature 中的 pixel_values 转换为 torch.Tensor
#                 pixel_values = torch.tensor(np.array(processed_patch["pixel_values"]))
#             else:
#                 raise KeyError(f"BatchFeature does not contain 'pixel_values': {processed_patch.keys()}")
#         else:
#             raise TypeError(f"Unexpected processor output type: {type(processed_patch)} at index {idx}")

#         processed_patches.append(pixel_values)

#     # 打印调试信息
#     #for i, p in enumerate(processed_patches):
#       #  print(f"Patch {i}: Type: {type(p)}, Shape: {getattr(p, 'shape', 'N/A')}")

#     # 断言所有元素是 Tensor
#     assert all(isinstance(p, torch.Tensor) for p in processed_patches), "All patches must be tensors."

#     image_patches_tensor = torch.stack(processed_patches)
    
#     return image_patches_tensor.squeeze(0, 2)


#已修改

def preprocess_phi_3_new(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    role_mapping = {"human": "user", "gpt": "assistant"}
    roles = ("<|user|>", "<|assistant|>")
    sep = "<s>"
    sep2 = "<|end|>"

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # TODO: add system prompt is there's not any in source.

        # Update key names
        for i, rnd in enumerate(source):
            if "from" in rnd:
                if rnd["from"] in ["human", "gpt"]:
                    rnd["role"] = role_mapping[rnd.pop("from")]
                else:
                    rnd["role"] = rnd.pop("from")
            if "value" in rnd:
                rnd["content"] = rnd.pop("value")
        # Apply chat template
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'system' %}{{ '<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        chat_conv = tokenizer.apply_chat_template(source, tokenize=False)
        chat_conv = chat_conv.replace(tokenizer.bos_token, '')

        conversations.append(chat_conv)

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    # assert conv.sep_style == conversation_lib.SeparatorStyle.PHI_3

    # Mask targets
    sep = roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
       # print(f"Total length before processing: {total_len}")

        rounds = conversation.split(sep2 + '\n')
        cur_len = 0  # No <bos> token.
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou += sep2 + '\n'
            if sep in rou:
                # assistant round
              #  print(f"Original rou for assistant: {rou}")  # 打印原始的 rou 信息
                round_ids = tokenizer(rou, max_length=max_len, truncation=True).input_ids
              #  print(f"Tokenized round_ids for assistant: {round_ids}")  # 打印分词后的 round_ids 信息
                sep2=roles[1]
                len_prefix=9
                if sep2 in tokenizer.get_vocab():
                   #print("<|assistant|> is in the vocabulary.")
                   role_prefix_ids = tokenizer(sep).input_ids  
                   len_prefix = len(role_prefix_ids)
                   #print(f"Round ids length: {len(round_ids)}, Role prefix length: {len_prefix}")
                
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif roles[0] in rou:
                # user round
                rou += sep
               # print(f"Original rou for user: {rou}")  # 打印原始的 rou 信息
                round_ids = tokenizer(rou, max_length=max_len, truncation=True).input_ids
               # print(f"Tokenized round_ids for user: {round_ids}")  # 打印分词后的 round_ids 信息
                rou_without_sep_length = len(tokenizer(rou[:-len(sep)], max_length=max_len, truncation=True).input_ids)
                round_len = len(round_ids)
                instruction_len = rou_without_sep_length
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            else:
                # system round
                #print(f"Original rou for system: {rou}")  # 打印原始的 rou 信息
                round_ids = tokenizer(rou, max_length=max_len, truncation=True).input_ids
               # print(f"Tokenized round_ids for system: {round_ids}")  # 打印分词后的 round_ids 信息
                round_len = len(round_ids)
                instruction_len = round_len
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
           # print(f"Current length after round {i}: {cur_len}")

        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len:  # The input_ids are truncated to this max length.
            if cur_len!= total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
      #  print(f"Final target length: {int(target.ne(tokenizer.pad_token_id).sum())}")
        
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template_name: Optional[str] = None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    
    if conv_template_name is not None and conv_template_name in conversation_lib.conv_templates.keys():
        # Use the specified preproseccing func.
        conv_template = conversation_lib.conv_templates[conv_template_name]
    else:
        conv_template = conversation_lib.default_conversation

    if conv_template.version.startswith("phi_3"):
        return preprocess_phi_3_new(sources, tokenizer)
    else:
        
        raise NotImplementedError
    

def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)

        padding = torch.full(
            (max_tokens - num_tokens,) + tuple(tensor.shape[1:]),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    if generator is not None:
        torch.manual_seed(42)
    megabatch_indices = torch.randperm(len(megabatches), generator=generator.manual_seed(42))
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)
