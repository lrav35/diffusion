from torch.utils.data import Dataset
from torch.optim import AdamW
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from transformers import PreTrainedTokenizer, CLIPTokenizer, CLIPTextModel
from pathlib import Path
from torchvision import transforms
from PIL import Image
from os import makedirs
import bitsandbytes as bnb
import itertools
import argparse

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)


class DreamBoothDataset(Dataset):
    """
    Dataset for training with Dreambooth. Code largely taken from diffusers examples.
    """
    def __init__(
        self,
        data_root: str,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        size=512, # resolution
    ) -> None:
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_image_path = list(Path(data_root).iterdir())
        self.num_images = len(self.instance_image_path)
        self.prompt = prompt
        self._length = self.num_images

        self.image_transforms_resize_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> dict:
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example
    

class PromptDataset(Dataset):
    def __init__(self, prompt: str, samples: int) -> None:
        self.prompt = prompt
        self.samples = samples

    def __len__(self) -> int:
        return self.samples
    
    def __getitem__(self, index: int) -> dict:
        example = {}
        example['prompt'] = self.prompt
        example['index'] = index
        return example
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="prompt for training data - needed for CLIP + diffusion combo",
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help="directory where the trainging data is stored",
    )
    args = parser.parse_args()


# !accelerate launch train_dreambooth_inpaint.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting"  \
#     --instance_data_dir="images/dog" \
#     --output_dir="stable-diffusion-inpainting-dog" \
#     --instance_prompt="a photo of a sks dog" \
#     --resolution=512 \
#     --mixed_precision="fp16" \
#     --train_batch_size=1 \
#     --learning_rate=5e-6 \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=0 \
#     --max_train_steps=500 \
#     --gradient_accumulation_steps=2 \
#     --gradient_checkpointing \
#     --train_text_encoder \
#     --seed="0"
    

# !accelerate launch fine_tune_dreambooth.py \
#     --prompt="insert prompt here"
#     --input_data_dir="images/"
    
def main():
    args = parse_args()

    model_name = "runwayml/stable-diffusion-inpainting"

    log_dir = Path("diffusion-inpainting")

    project_config = ProjectConfiguration(
        project_dir=log_dir, logging_dir=log_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=project_config
    )

    set_seed("42")

    # Handle the repository creation
    if accelerator.is_main_process:
        makedirs(log_dir, exist_ok=True) 

    # Load Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # disables grad descent for vae and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # enable gradient checkpointing for unet and text encoder
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

    # use for GPU with 16GB VRAM... if not, use AdamW
    optimizer_class = bnb.optim.Adam8bit
    # optimizer_class = AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )

    noise_sched = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        data_root=args.input_data_dir,
        prompt=args.prompt,
        tokenizer=tokenizer
    )

