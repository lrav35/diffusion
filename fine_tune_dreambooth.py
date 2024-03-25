from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from pathlib import Path
from torchvision import transforms
from PIL import Image


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
