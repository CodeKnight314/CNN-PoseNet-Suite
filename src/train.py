from src.dataset import load_data
from tqdm import tqdm
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.model import *
import numpy as np
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "pose_net_g":
        logger.info("Loading PoseNetG")
        return PoseNetG().to(device)
    elif model_name == "pose_net_r18":
        logger.info("Loading PoseNetR18")
        return PoseNetR18().to(device)
    elif model_name == "pose_net_r34":
        logger.info("Loading PoseNetR34")
        return PoseNetR34().to(device)
    elif model_name == "pose_net_effb0":
        logger.info("Loading PoseNetEffB0")
        return PoseNetEffB0().to(device)
    elif model_name == "pose_net_effb1":
        logger.info("Loading PoseNetEffB1")
        return PoseNetEffB1().to(device)
    elif model_name == "pose_net_mobv3l":
        logger.info("Loading PoseNetMobV3L")
        return PoseNetMobV3L().to(device)
    elif model_name == "pose_net_mobv3s":
        logger.info("Loading PoseNetMobV3S")
        return PoseNetMobV3S().to(device)
    else:
        logger.error(f"Invalid model: {model_name}")
        raise ValueError(f"Invalid model: {model_name}")
    
def train_step(model: nn.Module, optimizer: optim.Optimizer, img: torch.Tensor, pose: torch.Tensor, beta: float):
    optimizer.zero_grad()
    prediction = model(img)
    translation_loss = F.mse_loss(prediction[:, :3], pose[:, :3])
    rotation_loss = F.mse_loss(prediction[:, 3:], pose[:, 3:])
    loss = translation_loss + beta * rotation_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def test_step(model: nn.Module, img: torch.Tensor, pose: torch.Tensor, beta: float):
    with torch.no_grad():
        prediction = model(img)
        translation_loss = F.mse_loss(prediction[:, :3], pose[:, :3])
        rotation_loss = F.mse_loss(prediction[:, 3:], pose[:, 3:])
        loss = translation_loss + beta * rotation_loss
        return loss.item()
    
def log_details(config: dict):
    logger.info(f"Model: {config['model']}")
    logger.info(f"Beta: {config['beta']}")
    logger.info(f"LR: {config['lr']}")
    logger.info(f"Weight decay: {config['weight_decay']}")
    logger.info(f"Momentum: {config['momentum']}")
    logger.info(f"Step size: {config['step_size']}")
    logger.info(f"Gamma: {config['gamma']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Num workers: {config['num_workers']}")
    logger.info(f"Epochs: {config['epochs']}")

def train(path: str, config_path: str, save_path: str, verbose: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    log_details(config)

    train_loader = load_data(path, "train", config["batch_size"], config["num_workers"])
    test_loader = load_data(path, "test", config["batch_size"], config["num_workers"])

    model = load_model(config["model"])
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_loss = float("inf")
    train_losses = []
    test_losses = []
    pbar = tqdm(range(1, config["epochs"] + 1))
    for epoch in pbar:
        epoch_train_loss = []
        epoch_test_loss = []
        
        model.train()
        for batch in train_loader:
            img, pose = batch
            img = img.to(device)
            pose = pose.to(device)
            
            loss = train_step(model, optimizer, img, pose, config["beta"])
            epoch_train_loss.append(loss)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                img, pose = batch
                img = img.to(device)
                pose = pose.to(device)
                loss = test_step(model, img, pose, config["beta"])
                epoch_test_loss.append(loss)

        scheduler.step()
        
        avg_train_loss = np.mean(epoch_train_loss)
        avg_test_loss = np.mean(epoch_test_loss)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            model.save(os.path.join(save_path, f"{config['model']}.pth"))
            if verbose:
                logger.info(f"Best loss @ {epoch}: {best_loss}. Saving model...")
        
        if epoch % config["save_interval"] == 0:
            model.save(os.path.join(save_path, f"{config['model']}_checkpoint.pth"))
            if verbose:
                logger.info(f"Saving model @ {epoch}: {avg_test_loss}")

        pbar.update(1)        
        pbar.set_postfix(train_loss=avg_train_loss, test_loss=avg_test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    train(args.path, args.config, args.save_path, args.verbose)