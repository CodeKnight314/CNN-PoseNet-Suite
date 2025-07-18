from src.train import load_model
import os
import argparse
import torch
from glob import glob 
from tqdm import tqdm
from typing import List
from PIL import Image
import numpy as np
from torchvision import transforms as T
from collections import deque
import re

def compute_mean_from_dirs(dirs: List[str]):
    img_paths = []
    for d in dirs:
        img_paths.extend(glob(os.path.join(d, "*.color.png")))
    
    imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
    pixels = np.concatenate([np.asarray(img).reshape(-1, 3) for img in imgs], axis=0)
    mean_rgb = np.mean(pixels, axis=0) / 255.0
    return torch.tensor(mean_rgb.tolist()).to(torch.float32).view(3, 1, 1)

def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    dot = np.dot(q1, q2)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result)
    
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

def extract_frame_number(filepath):
    """Extract frame number from filepath for sorting."""
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    return 0

class PoseSmoothing:
    def __init__(self, translation_window=5, rotation_weight=0.8):
        """
        Initialize pose smoothing filters.
        
        Args:
            translation_window: Number of frames to use for moving average
            rotation_weight: Weight for SLERP interpolation (0-1, higher = more smoothing)
        """
        self.translation_window = translation_window
        self.rotation_weight = rotation_weight
        self.translation_buffer = deque(maxlen=translation_window)
        self.last_rotation = None
        
    def smooth_translation(self, translation):
        """Apply moving average filter to translation."""
        self.translation_buffer.append(translation.copy())
        return np.mean(self.translation_buffer, axis=0)
    
    def smooth_rotation(self, rotation):
        """Apply SLERP-based smoothing to rotation quaternion."""
        if self.last_rotation is None:
            self.last_rotation = rotation.copy()
            return rotation
        
        smoothed = slerp(self.last_rotation, rotation, 1.0 - self.rotation_weight)
        self.last_rotation = smoothed.copy()
        return smoothed

def vectors_to_transformation_matrix(translation: torch.Tensor, rotation: torch.Tensor):
    qx, qy, qz, qw = rotation[0].item(), rotation[1].item(), rotation[2].item(), rotation[3].item()
    tx, ty, tz = translation[0].item(), translation[1].item(), translation[2].item()
    R = np.array([
      [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
      [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
      [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T

def load_gt_pose(gt_path: str) -> np.ndarray:
    """Load ground truth pose from 4x4 transformation matrix file."""
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        row = [float(val) for val in line.strip().split()]
        matrix.append(row)
    
    return np.array(matrix)

def compute_translation_error(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    """Compute Euclidean distance between predicted and ground truth translations."""
    pred_translation = pred_matrix[:3, 3]
    gt_translation = gt_matrix[:3, 3]
    return np.linalg.norm(pred_translation - gt_translation)

def compute_quaternion_error(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    """Compute quaternion error between predicted and ground truth rotations."""
    pred_R = pred_matrix[:3, :3]
    gt_R = gt_matrix[:3, :3]
    
    def rotation_matrix_to_quaternion(R):
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    pred_q = rotation_matrix_to_quaternion(pred_R)
    gt_q = rotation_matrix_to_quaternion(gt_R)
    
    pred_q = pred_q / np.linalg.norm(pred_q)
    gt_q = gt_q / np.linalg.norm(gt_q)
    
    dot_product = np.abs(np.dot(pred_q, gt_q))
    dot_product = np.clip(dot_product, 0.0, 1.0)
    angular_error = 2 * np.arccos(dot_product) * 180 / np.pi
    
    return angular_error

def matrix_to_txt(transformation_matrix: np.ndarray, path: str):
    M = np.asarray(transformation_matrix, dtype=float)

    with open(path, 'w') as f:
        for row in M:
            line = ' '.join(f'{val:.6f}' for val in row)
            f.write(line + '\n')

def inference(args):
    model = load_model(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.w: 
        model.load_state_dict(torch.load(args.w, map_location=device))
    
    img_dir = glob(os.path.join(args.img_dir, "*.color.png"))
    img_dir.sort(key=extract_frame_number)
   
    if args.mdir:
        mean_rgb = compute_mean_from_dirs(args.mdir)
    else:
        mean_rgb = compute_mean_from_dirs([args.img_dir])

    transforms = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    pose_smoother = PoseSmoothing(
        translation_window=getattr(args, 'translation_window', 5),
        rotation_weight=getattr(args, 'rotation_weight', 0.8)
    )
    
    translation_errors = []
    quaternion_errors = []
    
    for img_path in tqdm(img_dir, desc="Processing frames in order"):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < h:
            new_w = 256
            new_h = int(h * new_w / w)
        else:
            new_h = 256
            new_w = int(w * new_h / h)

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = transforms(img)
        img = img - mean_rgb
        img = img.to(device)
        
        with torch.no_grad():
            prediction = model(img.unsqueeze(0)).squeeze(0)
            translation = prediction[:3].cpu().numpy()
            rotation = prediction[3:].cpu().numpy()
            rotation = rotation / np.linalg.norm(rotation)
            
        smoothed_translation = pose_smoother.smooth_translation(translation)
        smoothed_rotation = pose_smoother.smooth_rotation(rotation)
        
        translation_tensor = torch.tensor(smoothed_translation)
        rotation_tensor = torch.tensor(smoothed_rotation)
        transformation_matrix = vectors_to_transformation_matrix(translation_tensor, rotation_tensor)
        
        filename = os.path.basename(img_path).replace(".color.png","")
        txt_path = os.path.join(args.img_dir, f"{filename}.pose.txt")
        
        matrix_to_txt(transformation_matrix, txt_path)
        
        if args.gtdir:
            gt_pose_path = os.path.join(args.gtdir, f"{filename}.pose.txt")
            if os.path.exists(gt_pose_path):
                gt_matrix = load_gt_pose(gt_pose_path)
                
                trans_error = compute_translation_error(transformation_matrix, gt_matrix)
                quat_error = compute_quaternion_error(transformation_matrix, gt_matrix)
                
                translation_errors.append(trans_error)
                quaternion_errors.append(quat_error)
                
            else:
                print(f"Warning: Ground truth pose file not found: {gt_pose_path}")
    
    if args.gtdir and translation_errors:
        print("\n=== Error Summary ===")
        print(f"Mean Translation Error: {np.mean(translation_errors):.6f}")
        print(f"Std Translation Error: {np.std(translation_errors):.6f}")
        print(f"Mean Quaternion Error: {np.mean(quaternion_errors):.6f} degrees")
        print(f"Std Quaternion Error: {np.std(quaternion_errors):.6f} degrees")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference to compute pose transforms for a set of images."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--img_dir", type=str, required=True,
        help="Directory containing .color.png images to process"
    )
    
    parser.add_argument(
        "--w", type=str,
        help="Weights for model for inference"
    )
    
    parser.add_argument(
        "--mdir", nargs="+", type=str,
        help="List of directories to use for mean RGB computation"
    )
    
    parser.add_argument(
        "--gtdir", type=str, default=None,
        help="Directory of ground truth poses for comparison"
    )
    
    parser.add_argument(
        "--translation_window", type=int, default=10,
        help="Window size for moving average filter on translation (default: 5)"
    )
    
    parser.add_argument(
        "--rotation_weight", type=float, default=0.8,
        help="Smoothing weight for SLERP rotation interpolation, 0-1 (default: 0.8)"
    )

    args = parser.parse_args()
    inference(args)