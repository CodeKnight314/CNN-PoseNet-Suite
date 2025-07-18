import numpy as np
from glob import glob
import os
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional
from io import BytesIO
from tqdm import tqdm

def load_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return img

def txt_to_matrix(txt_path: str):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    matrix = np.array([[float(x) for x in line.split()] for line in lines])
    return matrix

def load_trajectories(path_to_trajectory: str, path_to_images: str):
    """
    Load trajectory with various coordinate system correction options.
    """
    images_files = sorted(glob(os.path.join(path_to_images, "*.color.png")))
    images = []
    trajectories_files = sorted(glob(os.path.join(path_to_trajectory, "*.txt")))
    trajectories = []
    
    for image_file, trajectory_file in tqdm(zip(images_files, trajectories_files), 
                                            total=len(images_files), 
                                            desc="Loading images and trajectories"):
        image_filename = os.path.basename(image_file).replace(".color.png", "")
        trajectory_filename = os.path.basename(trajectory_file).replace(".pose.txt", "")
        if image_filename != trajectory_filename:
            raise ValueError(f"Image and trajectory filenames do not match: {image_filename} != {trajectory_filename}")
        
        trajectory = txt_to_matrix(trajectory_file)
        trajectories.append(trajectory)
        image = load_image(image_file)
        images.append(image)

    return images, trajectories

class TrajectoryPlotter:
    def __init__(self, poses: List[np.array], axis_length=0.05, title_prefix=""):
        self.poses = poses
        self.axis_length = axis_length
        self.title_prefix = title_prefix
        
        self.all_positions = np.array([pose[:3, 3] for pose in poses])
        
        self.bounds = self._compute_bounds()
        
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def _compute_bounds(self):
        """Compute reasonable bounds for the plot based on trajectory."""
        positions = self.all_positions
        
        if len(positions) == 0:
            return {'x': (-1, 1), 'y': (-1, 1), 'z': (-1, 1)}
        
        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        
        ranges = maxs - mins
        padding = np.maximum(ranges * 0.1, 0.1)
        
        bounds = {
            'x': (mins[0] - padding[0], maxs[0] + padding[0]),
            'y': (mins[1] - padding[1], maxs[1] + padding[1]),
            'z': (mins[2] - padding[2], maxs[2] + padding[2])
        }
        
        return bounds
        
    def draw_frame(self, current_idx: int):
        self.ax.clear()
        
        self.ax.set_xlim(*self.bounds['x'])
        self.ax.set_ylim(*self.bounds['y'])
        self.ax.set_zlim(*self.bounds['z'])
        
        title = f"{self.title_prefix}Frame {current_idx}"
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        if current_idx > 0:
            positions = self.all_positions[:current_idx+1]
            self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                        color='black', linewidth=2, alpha=0.7)
        
        if current_idx < len(self.poses):
            current_pose = self.poses[current_idx]
            t = current_pose[:3, 3]
            R = current_pose[:3, :3]
            
            x_axis = R[:, 0] * self.axis_length
            y_axis = R[:, 1] * self.axis_length
            z_axis = R[:, 2] * self.axis_length
            
            self.ax.quiver(*t, *x_axis, color='r', linewidth=3, arrow_length_ratio=0.3, label='X')
            self.ax.quiver(*t, *y_axis, color='g', linewidth=3, arrow_length_ratio=0.3, label='Y')
            self.ax.quiver(*t, *z_axis, color='b', linewidth=3, arrow_length_ratio=0.3, label='Z')
            
            self.ax.scatter(*t, color='red', s=50, alpha=0.8)
        
        if current_idx == 0:
            self.ax.legend(loc='upper right')
        
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plot_img = np.array(Image.open(buf))
        buf.close()
        
        return plot_img
    
    def close(self):
        plt.close(self.fig)

def create_comparison_video(gt_poses, pred_poses, images, output_path, fps=10):
    """Create a side-by-side video comparing ground truth and predicted trajectories."""
    img_array = np.array(images[0])
    height, width, _ = img_array.shape
    
    gt_plotter = TrajectoryPlotter(gt_poses, title_prefix="GT: ")
    pred_plotter = TrajectoryPlotter(pred_poses, title_prefix="Pred: ")
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 3, height))
    
    for i in tqdm(range(len(images)), total=len(images), desc="Creating comparison video"):
        img_array = np.array(images[i])
        
        gt_plot = gt_plotter.draw_frame(i)
        pred_plot = pred_plotter.draw_frame(i)
        
        gt_plot = cv2.resize(gt_plot, (width, height))
        pred_plot = cv2.resize(pred_plot, (width, height))
        
        if gt_plot.shape[2] == 4:
            gt_plot = gt_plot[:, :, :3]
        if pred_plot.shape[2] == 4:
            pred_plot = pred_plot[:, :, :3]
        
        combined = np.hstack([img_array, gt_plot, pred_plot])
        out.write(combined)

    gt_plotter.close()
    pred_plotter.close()
    out.release()
    print(f"Comparison video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced trajectory visualization with debugging")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth poses directory")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted poses directory")
    parser.add_argument("--img_dir", type=str, required=True, help="Images directory")
    parser.add_argument("--output", type=str, default="trajectory_comparison.mp4", help="Output video path")
    
    args = parser.parse_args()
    
    print("Loading ground truth trajectory...")
    gt_images, gt_poses = load_trajectories(args.gt_dir, args.img_dir)
    
    print("Loading predicted trajectory...")
    pred_images, pred_poses = load_trajectories(args.pred_dir, args.img_dir)
    
    print(f"\nCreating comparison video...")
    create_comparison_video(gt_poses, pred_poses, gt_images, args.output)