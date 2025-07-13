import matplotlib.pyplot as plt 
from src.model import PoseNetG, PoseNetR18, PoseNetR34, PoseNetEffB0, PoseNetEffB1, PoseNetMobV3L, PoseNetMobV3S
import torch.nn as nn
import numpy as np
import torch 
import time
from tqdm import tqdm

def get_model_size(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_time_per_batch(model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    IMG_SIZE = 224
    TRIALS = 100

    runtime = 0.0
    model.to(device)
    model.eval()
    for _ in tqdm(range(TRIALS), desc="Trials", position=1, leave=False):
        img = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)
        start_time = time.time()
        with torch.no_grad():
            model(img)
        runtime += time.time() - start_time
    return runtime / TRIALS

def profile_models():
    models = [PoseNetG(), PoseNetR18(), PoseNetR34(), PoseNetEffB0(), PoseNetEffB1(), PoseNetMobV3L(), PoseNetMobV3S()]
    model_names = ['PoseNetG', 'PoseNetR18', 'PoseNetR34', 'PoseNetEffB0', 'PoseNetEffB1', 'PoseNetMobV3L', 'PoseNetMobV3S']
    models_params = []
    models_time = []

    for model in tqdm(models, desc="Profiling models", position=0):
        models_params.append(get_model_size(model))
        models_time.append(get_time_per_batch(model))

    min_params = min(models_params)
    marker_sizes = [50 * (params / min_params) for params in models_params]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

    plt.figure(figsize=(12, 8))
    for i, (params, time, size, color, name) in enumerate(zip(models_params, models_time, marker_sizes, colors, model_names)):
        plt.scatter(params, time, s=size, c=color, alpha=0.7, label=name)
    
    plt.xlabel('Model Size (Num Parameters)')
    plt.xscale('log')
    plt.ylabel('Time per batch (s)')
    plt.yscale('log')
    plt.title('Time per batch vs Model Size for different models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('models_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    profile_models()
