from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
from PIL import Image
from torchvision import transforms
import numpy as np

class PoseNetDataset(Dataset):
    def __init__(self, data_path: str, split: str):
        self.data_path = data_path
        self.data = json.load(open(os.path.join(data_path, f"{split}.json")))
        self.img_paths = list(self.data.keys())
        self.poses = list(self.data.values()) 

        self.transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.mean_rgb = torch.tensor(self.compute_per_scene_mean()).to(torch.float32).view(3, 1, 1)

    def compute_per_scene_mean(self):
        imgs = [Image.open(img_path).convert("RGB") for img_path in self.img_paths]
        pixels = np.concatenate([np.asarray(img).reshape(-1, 3) for img in imgs], axis=0)
        mean_rgb = np.mean(pixels, axis=0) / 255.0
        return mean_rgb.tolist()

    def __len__(self):
        return len(self.img_paths)
    
    def transform(self, img_path: str):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < h:
            new_w = 256
            new_h = int(h * new_w / w)
        else:
            new_h = 256
            new_w = int(w * new_h / h)

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = self.transforms(img)
        img = img - self.mean_rgb
        return img
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        pose = self.poses[idx]
        pose = torch.tensor(pose, dtype=torch.float32)
        img = self.transform(img_path)

        return img, pose
    
def load_data(data_path: str, split: str, batch_size: int, num_workers: int=os.cpu_count()//2):
    dataset = PoseNetDataset(data_path, split)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)