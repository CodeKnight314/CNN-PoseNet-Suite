import matplotlib.pyplot as plt
import csv

def load_csv():
    models = ["pose_net_effb0", "pose_net_effb1", "pose_net_g", "pose_net_r34", "pose_net_r18", "pose_net_mobv3l", "pose_net_mobv3s"] 
    csv_paths = [f"resources/csv/{model}.csv" for model in models]

    scene = {
        "chess": {}, 
        "fire": {},
        "heads": {},
        "office": {},
        "pumpkin": {},
        "stairs": {},
    }

    for csv_path in csv_paths:
        model_name = csv_path.split("/")[-1].replace(".csv", "")
        
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            
            header = next(reader)
            
            scene_columns = {}
            for i, col_name in enumerate(header):
                if "epoch/test_loss" in col_name and "__MIN" not in col_name and "__MAX" not in col_name:
                    scene_name = col_name.split("_")[0]
                    if scene_name in scene:
                        scene_columns[scene_name] = i
                        if model_name not in scene[scene_name]:
                            scene[scene_name][model_name] = []
            
            for row in reader:
                for scene_name, col_index in scene_columns.items():
                    if col_index < len(row) and row[col_index]:
                        try:
                            scene[scene_name][model_name].append(float(row[col_index]))
                        except ValueError:
                            pass

    return scene 

def plot_graphs(scene: dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (scene_name, model_data) in enumerate(scene.items()):
        ax = axes[i]
        
        for model_name, losses in model_data.items():
            if losses:
                ax.plot(losses, label=model_name)
        
        ax.set_title(scene_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig("resources/graphs/per_scene_losses.png")
    plt.close()

if __name__ == "__main__":
    scene = load_csv()
    plot_graphs(scene)
