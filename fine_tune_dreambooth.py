from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import PreTrainedTokenizer, CLIPTokenizer, CLIPTextModel
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageDraw
from os import makedirs, path
from tqdm.auto import tqdm
import bitsandbytes as bnb
import itertools
import argparse
import random
import numpy as np
import math


from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler

logger = get_logger(__name__)

def random_mask(shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(shape[0] * ratio)), random.randint(0, int(shape[1] * ratio)))
    limits = (shape[0] - size[0] // 2, shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image



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
        instance_image = Image.open(self.instance_image_path[index % self.num_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["input_prompt_ids"] = self.tokenizer(
            self.prompt,
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
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="maximum number of training steps in any given run",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=True,
        help="batch size of training run",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        help="we need to use fp16 bc we are VRAM bound training on colab T4 GPUs"
    )
    args = parser.parse_args()
    
    return args
    
def main():
    """
    training code for dreambooth diffusion model. Code largely taken from diffusers examples.
    """
    args = parse_args()

    model_name = "runwayml/stable-diffusion-inpainting"

    log_dir = Path("diffusion-inpainting")

    project_config = ProjectConfiguration(
        project_dir=log_dir, logging_dir=log_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config
    )

    set_seed(42)

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

    def _collate_fn(examples):
        input_ids = [example["input_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_images"]
            # generate a random mask
            mask = random_mask(pil_image.size, 1, False)
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=_collate_fn
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps*accelerator.num_processes
    )

    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    gradient_accumulation_steps=2
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args)) 

    #### TRAIN #####
    total_batch_size = args.train_batch_size * accelerator.num_processes * 2 # batch * processes * grad accum

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {2}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # images -> latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # masks -> latent space
                latent_masks = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                latent_masks = latent_masks * vae.config.scaling_factor

                masks = batch["masks"]
                # resize masks to shape of latents as we concatentate the masks to the latents
                new_res = 512 // 8
                mask = torch.stack(
                    [
                        F.interpolate(mask, size=(new_res, new_res)) for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, new_res, new_res)

                # sample noise to add to latents (forward pass)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # sample random timestep (forward pass)
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # add this random noise to our latents
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # concat the noisy latents with the mask and unmasked latents
                latent_model_input = torch.cat([noisy_latents, mask, latent_masks], dim=1)

                # get text embedding for conditioning (CLIP)
                encoder_h_states = text_encoder(batch["input_ids"])[0]

                # make noise residual prediction
                noise_pred = unet(latent_model_input, timesteps, encoder_h_states).sample

                # Get the target for loss depending on the prediction type
                if noise_sched.config.prediction_type == "epsilon":
                    target = noise
                elif noise_sched.config.prediction_type == "v_prediction":
                    target = noise_sched.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_sched.config.prediction_type}")
                
                # LOSS
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 500 == 0:
                    if accelerator.is_main_process:
                        save_path = path.join(log_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
    
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(log_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
