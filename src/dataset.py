from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
from PIL import Image
from torchvision import transforms

class PoseNetDataset(Dataset):
    def __init__(self, data_path: str, split: str):
        self.data_path = data_path
        self.data = json.load(open(os.path.join(data_path, f"{split}.json")))
        self.img_paths = list(self.data.keys())
        self.poses = list(self.data.values()) 

        self.transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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