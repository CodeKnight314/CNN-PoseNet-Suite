import os
import numpy as np 
from glob import glob
import json 
import argparse
from tqdm import tqdm

def save_data(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f)

def matrix_to_posenet_vector(T: np.ndarray) -> np.ndarray:
    """
    Converts a 4x4 transformation matrix into a 7D PoseNet pose vector.
    Returns [x, y, z, qx, qy, qz, qw]
    """
    assert T.shape == (4, 4), "Input must be a 4x4 transformation matrix."

    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    R = T[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2 
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2 
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2 
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2 
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    quat = np.array([qx, qy, qz, qw])
    quat /= np.linalg.norm(quat)

    return np.array([x, y, z, *quat])

def txt_to_matrix(txt_path: str):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    matrix = np.array([[float(x) for x in line.split()] for line in lines])
    return matrix

def read_split(path: str):
    if not os.path.exists(path):
        raise ValueError(f"[ERROR] Invalid path for {path}")
    
    with open(path, 'r') as f: 
        lines = f.readlines()
    splits = [f"seq-0{s.strip()[-1]}" for s in lines]
    return splits

def handle_scene(path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    train_split = read_split(os.path.join(path, "TrainSplit.txt"))
    test_split = read_split(os.path.join(path, "TestSplit.txt"))

    train_data = {}
    test_data = {}
    for split in tqdm(train_split, desc="Processing train data"):
        seq_path = os.path.join(path, split)
        
        rgb_images = glob(os.path.join(seq_path, "*.color.png"))
        labels = glob(os.path.join(seq_path, "*.pose.txt"))

        for rgb_image, label in zip(rgb_images, labels):
            label_matrix = txt_to_matrix(label)
            pose = matrix_to_posenet_vector(label_matrix)
            train_data[rgb_image] = pose.tolist()

    for split in tqdm(test_split, desc="Processing test data"):
        seq_path = os.path.join(path, split)
        
        rgb_images = glob(os.path.join(seq_path, "*.color.png"))
        labels = glob(os.path.join(seq_path, "*.pose.txt"))

        for rgb_image, label in zip(rgb_images, labels):
            label_matrix = txt_to_matrix(label)
            pose = matrix_to_posenet_vector(label_matrix)
            test_data[rgb_image] = pose.tolist()

    save_data(train_data, os.path.join(output_path, "train.json"))
    save_data(test_data, os.path.join(output_path, "test.json"))

def main(args):
    for scene in os.listdir(args.p):
        scene_path = os.path.join(args.p, scene)
        handle_scene(scene_path, os.path.join(args.o, scene))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--o", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    main(args)