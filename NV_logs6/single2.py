import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_filename(filename):
    parts = filename.split('_')
    method = parts[0]
    weight_decay = parts[2]
    lr = parts[4]
    dataset = parts[6]  # This represents the dataset size
    try:
        beta = parts[8]
        # Remove the .csv extension
        beta=beta.replace('.csv', '')
    except:
        beta = "0.1"
    return method, weight_decay, lr, beta, dataset

def find_csv_files(base_path, expert_dataset):
    csv_files = []
    path = os.path.join(base_path, expert_dataset)
    if os.path.exists(path):
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} csv files in {path}")
    return csv_files

def read_csv_data(csv_files):
    data = {}
    datasets = set()
    for file in csv_files:
        df = pd.read_csv(file)
        method, weight_decay, lr, beta, dataset = parse_filename(os.path.basename(file))
        if method not in ["CPO", "DPOfree"]:
            continue
        key = (method, beta, dataset, weight_decay)
        if key not in data:
            data[key] = []
        data[key].append(df)
        datasets.add(dataset)
    return data, sorted(list(datasets))

def plot_returns(all_data, return_type, tasks, datasets, name):
    fig, axs = plt.subplots(len(tasks), len(datasets), figsize=(80, 15), sharex=True)
    
    total_timesteps = 100000
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    
    task_scores = {
        "Walker2d-v2": {"expert": 4592.3, "random": -0.48},
        "HalfCheetah-v2": {"expert": 12135.0, "random": -278.6},
        "Hopper-v2": {"expert": 3234.3, "random": -19.5}
    }
    
    for task_idx, task in enumerate(tasks):
        expert_score = task_scores[task]["expert"]
        random_score = task_scores[task]["random"]
        
        # Set y limits for this task
        y_min = random_score - 500
        y_max = expert_score + 500
        
        for dataset_idx, dataset in enumerate(datasets):
            ax = axs[task_idx, dataset_idx]
            
            if task not in all_data:
                ax.text(0.5, 0.5, f"No data for {task}", ha='center', va='center')
                continue
            
            color_idx = 0
            for method in ["DPOfree", "CPO"]:
                for beta in ["0.1", "0.2", "0.5", "0.8", "1.0"]:
                    for weight_decay in ["0e+00", "1e-03"]:
                        key = (method, beta, dataset, weight_decay)
                        if key in all_data[task]:
                            dfs = all_data[task][key]
                            averaged_df = pd.concat(dfs).groupby(level=0).mean()
                            
                            returns = averaged_df[return_type]
                            eval_freq = total_timesteps // len(returns)
                            steps = np.arange(0, total_timesteps, eval_freq)[:len(returns)]
                            
                            label = f'{method} (Beta={beta} WD={weight_decay})'
                            ax.plot(steps, returns, label=label, color=colors[color_idx])
                            color_idx += 1
            
            # Draw expert and random lines
            ax.axhline(y=expert_score, color='r', linestyle='--', label='Expert')
            ax.axhline(y=random_score, color='g', linestyle='--', label='Random')
            
            # Set y limits for this task
            ax.set_ylim(y_min, y_max)
            
            ax.set_title(f'{task} - Dataset Size {dataset}', fontsize=10)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Return')
            ax.grid(True)
            ax.legend(loc='lower left', fontsize='x-small')
            ax.set_xlim(0, total_timesteps)
    
    fig.suptitle(f'{return_type.capitalize()} Returns for Different Tasks and Dataset Sizes', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}_{return_type}', bbox_inches='tight', dpi=300)
    print(f"Saved {name}_{return_type}.png")
    plt.close()

def main():
    tasks = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2']
    expert_dataset = "DMILSetting3"
    
    all_data = {}
    datasets = set()
    for task in tasks:
        base_path = f'logs/{task}'
        csv_files = find_csv_files(base_path, expert_dataset)
        task_data, task_datasets = read_csv_data(csv_files)
        all_data[task] = task_data
        datasets.update(task_datasets)
    
    datasets = sorted(list(datasets))
    
    # Debug print
    for task, data in all_data.items():
        print(f"Task: {task}")
        for key, runs in data.items():
            print(f"    Method: {key[0]}, Beta: {key[1]}, Dataset Size: {key[2]}, Number of runs: {len(runs)}")
    
    # Plot returns
    plot_returns(all_data, 'deterministic_return', tasks, datasets, "multi_dataset_size")
    
    print("Visualization completed. Check the current directory for output images.")

if __name__ == "__main__":
    main()