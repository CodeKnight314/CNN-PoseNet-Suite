import numpy as np
from glob import glob
import os
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
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

def load_trajectory(path_to_trajectory: str, path_to_images: str):
    images_files = sorted(glob(os.path.join(path_to_images, "*.color.png")))
    images = []
    trajectories_files = sorted(glob(os.path.join(path_to_trajectory, "*.txt")))
    trajectories = []
    for image_file, trajectory_file in tqdm(zip(images_files, trajectories_files), total=len(images_files), desc="Loading images and trajectories"):
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
    def __init__(self, poses: List[np.array], axis_length=0.05):
        self.poses = poses
        self.axis_length = axis_length
        
        self.all_positions = np.array([pose[:3, 3] for pose in poses])
        
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.axis('on')
        
    def draw_frame(self, current_idx: int):
        self.ax.clear()
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_title(f"Frame {current_idx}")
        self.ax.axis('on')
        
        if current_idx > 0:
            positions = self.all_positions[:current_idx+1]
            self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='black', linewidth=2)
        
        if current_idx < len(self.poses):
            current_pose = self.poses[current_idx]
            t = current_pose[:3, 3]
            R = current_pose[:3, :3]
            
            x_axis = R[:, 0] * self.axis_length
            y_axis = R[:, 1] * self.axis_length
            z_axis = R[:, 2] * self.axis_length
            
            self.ax.quiver(*t, *x_axis, color='r', linewidth=3, arrow_length_ratio=0.3)
            self.ax.quiver(*t, *y_axis, color='g', linewidth=3, arrow_length_ratio=0.3)
            self.ax.quiver(*t, *z_axis, color='b', linewidth=3, arrow_length_ratio=0.3)
        
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plot_img = np.array(Image.open(buf))
        buf.close()
        
        return plot_img
    
    def close(self):
        plt.close(self.fig)

def draw_3d_plot(poses: List[np.array], current_idx: int, axis_length=0.05):
    plotter = TrajectoryPlotter(poses, axis_length)
    img = plotter.draw_frame(current_idx)
    plotter.close()
    return img

def create_side_by_side_video(poses, images, output_path, fps=10):
    img_array = np.array(images[0])
    height, width, _ = img_array.shape
    
    plotter = TrajectoryPlotter(poses)
    
    plot_img = plotter.draw_frame(0)
    plot_img = cv2.resize(plot_img, (width, height))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    for i in tqdm(range(len(images)), total=len(images), desc="Creating side-by-side video"):
        img_array = np.array(images[i])
        
        plot_img = plotter.draw_frame(i)
        plot_img = cv2.resize(plot_img, (width, height))
        
        if plot_img.shape[2] == 4:
            plot_img = plot_img[:, :, :3]
        
        combined = np.hstack([img_array, plot_img])
        out.write(combined)

    plotter.close()
    out.release()
    print(f"Saved video to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_trajectory", type=str, required=True, help="Path to the trajectory files")
    parser.add_argument("--path_to_images", type=str, required=True, help="Path to the images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output video")
    args = parser.parse_args()
    images, trajectories = load_trajectory(args.path_to_trajectory, args.path_to_images)
    create_side_by_side_video(trajectories, images, args.output_path)