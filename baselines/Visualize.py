import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_filename(filename):
    parts = filename.split('_')
    method = parts[0]
    params = {}
    reject_from = None
    i = 1
    while i < len(parts):
        if parts[i] == 'load':
            params['load_freq'] = parts[i+2]
            i += 3
        elif parts[i] == 'random':
            reject_from_parts = []
            i += 1
            while i < len(parts) and parts[i] not in ['weight', 'beta', 'eta', 'gamma', 'Lambda']:
                reject_from_parts.append(parts[i])
                i += 1
            reject_from = '_'.join(reject_from_parts)
        elif parts[i] == 'weight':
            params['weight_decay'] = parts[i+2]
            i += 3
        elif parts[i] == 'noise':
            params['noise'] = parts[i+1]
            i += 2
        elif parts[i] in ['beta', 'eta', 'gamma', 'Lambda']:
            params[parts[i]] = parts[i+1]
            i += 2
        else:
            i += 1
    return method, params, reject_from

def find_csv_files(base_path, expert_datasets):
    csv_files = {}
    for dataset in expert_datasets:
        path = os.path.join(base_path, dataset)
        if os.path.exists(path):
            csv_files[dataset] = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    return csv_files

def read_csv_data(csv_files):
    data = {}
    for dataset, files in csv_files.items():
        for file in files:
            df = pd.read_csv(file)
            method, params, reject_from = parse_filename(os.path.basename(file))
            key = (method, frozenset(params.items()), reject_from)
            if reject_from not in ["policy"]:
                continue
            if key not in data:
                data[key] = {}
            data[key][dataset] = df
    return data

def normalize_returns(returns, expert_score, random_score):
    return (returns - random_score) / (expert_score - random_score)

def select_best_methods(data, return_type,filter=None):
    best_methods = {}
    for (method, params, reject_from), datasets in data.items():
        if filter is not None and method not in filter:
            continue
        for dataset, df in datasets.items():
            max_return = df[return_type].max()
            if method not in best_methods or max_return > best_methods[method]['max_return']:
                best_methods[method] = {
                    'params': params,
                    'reject_from': reject_from,
                    'max_return': max_return,
                    'data': datasets
                }
    return best_methods
def select_all_methods(data, return_type,filter=None):
    i=0
    all_methods = {}
    for (method, params, reject_from), datasets in data.items():
        if filter is not None and method not in filter:
            continue
        i+=1
        for dataset, df in datasets.items():
            all_methods[method+f"{i}"] = {
                'params': params,
                'reject_from': reject_from,
                'data': datasets
            }
    return all_methods
def plot_returns(data, return_type, is_normalized=False,filters=None,params=False,expert_dataset=None,task=None):
    # params=True
    # filters=["CPO"]
    best_methods = select_best_methods(data, return_type, filter=filters)
    # best_methods = select_all_methods(data, return_type, filter=filters)
    
    fig, axs = plt.subplots(4, 1, figsize=(24, 40), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(best_methods)))
    
    if task == "logs/Walker2d-v2":
        # expert_score = 4924.278
        # random_score = 91.524
        expert_score=4592.3
        random_score=-0.48
    elif task == "logs/HalfCheetah-v2":
        # expert_score = 10656.426
        # random_score = -288.797
        expert_score=12135.0
        random_score=-278.6
    elif task == "logs/Ant-v2":
        expert_score = 4778.389
        random_score = -338.064
    elif task == "logs/Hopper-v2":
        # expert_score = 3607.890
        # random_score = 832.351
        expert_score=3234.3
        random_score=-19.5
    
    # expert_datasets = ['halfcheetah-100', 'halfcheetah-10', 'halfcheetah-5', 'halfcheetah-2']
    baseline_scores = {
        'halfcheetah-100': 0.9334,
        'halfcheetah-10': 0.9269,
        'halfcheetah-5': 0.9018,
        'halfcheetah-2': 0.7687,
        "walker2d-100": 1.0765,
        "walker2d-10": 1.0762,
        "walker2d-5": 1.0789,
        "walker2d-2": 1.0555,
        "hopper-2": 1.0851,
        "hopper-5": 1.1114,
        "hopper-10": 1.1156,
        "hopper-100": 1.1022,
    }
    
    total_timesteps = 200000
    
    for idx, dataset in enumerate(expert_dataset):
        ax = axs[idx]
        for (method, method_data), color in zip(best_methods.items(), colors):
            if dataset in method_data['data']:
                df = method_data['data'][dataset]
                returns = df[return_type]
                
                # Calculate evaluation frequency based on data length
                eval_freq = total_timesteps // len(returns)
                
                steps = np.arange(0, total_timesteps, eval_freq)[:len(returns)]
                
                if is_normalized:
                    returns = normalize_returns(returns, expert_score, random_score)
                if params:
                    label = f'{method} ({method_data["reject_from"]}, freq={eval_freq}, {method_data["params"]})'
                else:

                    label = f'{method} ({method_data["reject_from"]}, freq={eval_freq})'
                ax.plot(steps, returns, color=color, label=label, marker='o', markersize=3)
        
        if is_normalized:
            baseline = baseline_scores[dataset]
            ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.4f})')
            ax.set_ylim(-0.2, 1.3)
        
        ax.set_title(f'{dataset}', fontsize=14)
        ax.set_xlabel('Timesteps' if idx == 3 else '')
        ax.set_ylabel('Normalized Return' if is_normalized else 'Return')
        ax.grid(True)
        ax.legend(loc='upper left', fontsize='small')
        ax.set_xlim(0, total_timesteps)
    
    title_prefix = "Normalized " if is_normalized else ""
    fig.suptitle(f'{title_prefix}{return_type.capitalize()} Returns for Best Performing Methods', fontsize=16)
    plt.tight_layout()
    filename_prefix = "normalized_" if is_normalized else ""
    plt.savefig(f'{filename_prefix}{return_type}_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
def summarize_experiments(data):
    experiments = {
        'random': set(),
        'policy': set(),
        'add_gaussian_noise_expert_act': set(),
        'add_noise_expert_act': set()
    }
    
    for (method, _, reject_from) in data.keys():
        print(f"{method} ({reject_from})")
        if reject_from == 'random':
            experiments['random'].add(method)
        elif reject_from == 'policy':
            experiments['policy'].add(method)
        elif reject_from == 'add_gaussian_noise_expert_act':
            experiments['add_gaussian_noise_expert_act'].add(method)
        elif reject_from == 'add_noise_expert_act':
            experiments['add_noise_expert_act'].add(method)
    
    print("\nExperiment Summary:")
    print("-------------------")
    print("Random rejection:")
    print(", ".join(sorted(experiments['random'])) if experiments['random'] else "None")
    print("\nUniform rejection (policy):")
    print(", ".join(sorted(experiments['policy'])) if experiments['policy'] else "None")
    print("\nGaussian noise rejection:")
    print(", ".join(sorted(experiments['add_gaussian_noise_expert_act'])) if experiments['add_gaussian_noise_expert_act'] else "None")
    print("\nNoise from model rejection:")
    print(", ".join(sorted(experiments['add_noise_expert_act'])) if experiments['add_noise_expert_act'] else "None")

def main():
    base_path = 'logs/HalfCheetah-v2'
    expert_datasets = [
        'halfcheetah-100',
        'halfcheetah-10',
        'halfcheetah-5',
        'halfcheetah-2'
    ]
    # base_path = 'logs/Walker2d-v2'
    # expert_datasets = [
    #     'walker2d-100',
    #     'walker2d-10',
    #     'walker2d-5',
    #     'walker2d-2'
    # ]
    # base_path="logs/Hopper-v2"
    # expert_datasets=[
    #     "hopper-100",
    #     "hopper-10",
    #     "hopper-5",
    #     "hopper-2"
    # ]

    csv_files = find_csv_files(base_path, expert_datasets)
    data = read_csv_data(csv_files)

    # Plot original returns
    plot_returns(data, 'deterministic_return',  expert_dataset=expert_datasets,task=base_path)
    plot_returns(data, 'stochastic_return',expert_dataset=expert_datasets,task=base_path)

    # Plot normalized returns
    plot_returns(data, 'deterministic_return', is_normalized=True,expert_dataset=expert_datasets,task=base_path)
    plot_returns(data, 'stochastic_return', is_normalized=True,expert_dataset=expert_datasets,task=base_path)

    print("Visualization completed. Check the current directory for output images.")

    # Summarize experiments
    summarize_experiments(data)

if __name__ == "__main__":
    main()