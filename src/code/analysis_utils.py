#########################
# imports
##########################
from math import ceil
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
import itertools
import json
import numpy as np
import pandas as pd
pd.set_option('compute.use_numexpr', True)                                                                                                                       
from code.models import HierarchicalBayesianModel
import torch
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pyprojroot import here
from matplotlib.colors import to_hex, to_rgba


from .config import Config
from .utils import init_dataset, collect_outputs_and_labels 
from .models import get_algorithmic_solutions, load_transformer, init_transformer
from .compression_utils import compress_array, compress_python_script

import scipy.optimize as optimize

# set up tqdm
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from IPython import get_ipython

from .models import HierarchicalBayesianModel, HierarchicalBayesianModelAblated

#########################
# general helper functions

def tqdm_func():
    if get_ipython() is None:
        return tqdm  # Running in a script
    else:
        return tqdm_notebook  # Running in Jupyter Notebook
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch, device=get_device()):
    """Move all tensors in a batch to the specified device.
    
    Args:
        batch: A dictionary that may contain tensors or nested dictionaries with tensors
        device: The target device
        
    Returns:
        The batch with all tensors moved to the target device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    else:
        return batch

def median_of_means(X, k=None, rng=None):
    """Compute the median of means of X.
    
    Args:
        X: array-like of shape (n_samples,)
        k: int, optional
        rng: numpy.random.Generator, optional
        
    Returns:
        float: The median of means of X.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    if k is None:
        k = np.sqrt(len(X))
    perm = rng.permutation(len(X))
    blocks = np.array_split(perm, k)
    means = [X[idx].mean() for idx in blocks]
    return np.median(means)

def total_variation_distance(p, q):
    """Compute total variation distance between two probability distributions"""
    p = p.flatten(end_dim=-2)
    q = q.flatten(end_dim=-2)
    return 0.5 * torch.sum(torch.abs(p - q), axis=-1).mean()

def hellinger_distance(p, q):
    """Compute Hellinger distance between two probability distributions"""
    p = p.flatten(end_dim=-2)
    q = q.flatten(end_dim=-2)
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, axis=-1).mean())

def batched_spearman_correlation(x, y, dim=-1):
    """
    Compute Spearman rank correlation in a fully vectorized batched manner.
    
    Args:
        x: Tensor of shape (..., n)
        y: Tensor of shape (..., n)
        dim: Dimension along which to compute correlation (default: -1)
        
    Returns:
        Spearman correlation coefficients with shape (...,)
    """
    # flatten all dimensions except the last one
    x = x.flatten(end_dim=-2)
    y = y.flatten(end_dim=-2)
    
    # Get ranks using double argsort trick
    ranks_x = torch.argsort(torch.argsort(x, dim=dim), dim=dim).float() + 1.0
    ranks_y = torch.argsort(torch.argsort(y, dim=dim), dim=dim).float() + 1.0
    
    # Center the ranks
    ranks_x_centered = ranks_x - ranks_x.mean(dim=dim, keepdim=True)
    ranks_y_centered = ranks_y - ranks_y.mean(dim=dim, keepdim=True)
    
    # Compute correlation
    covariance = (ranks_x_centered * ranks_y_centered).sum(dim=dim)
    x_std = torch.sqrt((ranks_x_centered ** 2).sum(dim=dim))
    y_std = torch.sqrt((ranks_y_centered ** 2).sum(dim=dim))
    
    return (covariance / (x_std * y_std + 1e-8)).mean()
    
#########################
# Setup
#########################

def make_all_exp_param_combinations(exp_params):
    """make all combinations of experiment parameters"""
    # Get only the iterable parameters we want to use
    param_mapping = {
        'num_dims_lst': 'num_dims',
        'num_tasks_lst': 'num_tasks',
        'context_length_lst': 'context_length',
        'mlp_expansion_factor_lst': 'mlp_expansion_factor',
        'save_steps': 'checkpoint'
    }
    
    # Extract values and rename parameters
    iterable_params = {}
    for old_name, new_name in param_mapping.items():
        if old_name in exp_params and isinstance(exp_params[old_name], (list, tuple)):
            iterable_params[new_name] = exp_params[old_name]
            
    # make all combinations of iterables
    all_exp_param_combinations = list(itertools.product(*iterable_params.values()))
    param_names = list(iterable_params.keys())
    
    # Create list of dictionaries for DataFrame creation
    param_dicts = []
    for combo in all_exp_param_combinations:
        param_dict = dict(zip(param_names, combo))
        param_dicts.append(param_dict)
    
    return param_dicts


def make_results_dfs(exp_params, cache_dir):
    """Make a transformer_df to save transformer evaluation results and algo_df to save results for algorithmic solution evaluations"""
    # Create parameter combinations
    param_dicts = make_all_exp_param_combinations(exp_params)

    # Create transformer DataFrame
    transformer_df = (
        pd.DataFrame(param_dicts)
        .sort_values(["num_dims", "num_tasks", "context_length"])
        .reset_index(drop=True)
        .reset_index(drop=True)
    )

    # iterate over rows of transformer_df and add to yaml_filepaths
    configs = []
    for index, row in tqdm_func()(transformer_df.iterrows(), total=transformer_df.shape[0]):
        config = Config(
            cache_dir=cache_dir,
            setting=exp_params["setting"],
            num_dims=row["num_dims"],
            num_tasks=row["num_tasks"],
            context_length=row["context_length"],
            mlp_expansion_factor=row["mlp_expansion_factor"],
            random_seed=exp_params["random_seed"],
            num_hidden_layers=exp_params["num_hidden_layers"],
            hidden_size=exp_params["hidden_size"],
            max_steps=exp_params["max_steps"],
            save_steps=exp_params["save_steps"],
            batch_size=exp_params["batch_size"],
            learning_rate=exp_params["learning_rate"],
            name_suffix=exp_params["name_suffix"],
        )
        configs.append(config)

    # add as column to transformer_df
    transformer_df["config"] = configs


    # Create algo DataFrame
    algo_df = (
        transformer_df.copy()
        # remove transformer-specific columns
        .drop(columns=["mlp_expansion_factor"])
        # drop duplicate rows based on [num_dims, num_tasks, context_length]
        .drop_duplicates(subset=["num_dims", "num_tasks", "context_length"]).reset_index(
            drop=True
        )
    )
    return transformer_df, algo_df

#########################
# Evaluation
#########################

# Prediction with transformers

def run_evaluation_transformer(transformer_df,
                               exp_params,
                               num_eval_sequences,
                               only_final_models=True,
                               load_saved_evaluation=True,
                               add_to_df=True,
                               random_transformer_baseline=False):
    """run full evaluation for transformers
    
    returns transformer_df with train and eval metrics and probs.
    """
    # create empty columns for metrics and probs
    transformer_df['train_metrics'] = None
    transformer_df['eval_metrics'] = None
    transformer_df['train_outputs'] = None
    transformer_df['eval_outputs'] = None
    if exp_params["setting"] == "classification":
        transformer_df['iwl_metrics'] = None
        transformer_df['iwl_outputs'] = None
    if random_transformer_baseline:
        transformer_df['train_outputs-random'] = None
        transformer_df['eval_outputs-random'] = None
        transformer_df['train_metrics-random'] = None
        transformer_df['eval_metrics-random'] = None
        if exp_params["setting"] == "classification":
            transformer_df['iwl_metrics-random'] = None
            transformer_df['iwl_outputs-random'] = None

    # iterate over rows of transformer_df
    for index, row in tqdm_func()(transformer_df.iterrows(), total=transformer_df.shape[0]):
        # skip if evaluation runs on only final models and checkpoint is not the final checkpoint
        if only_final_models and row['checkpoint'] != exp_params['max_steps']:
            continue

        config = row['config']
        model_path = os.path.join(config.output_dir, f"checkpoint-{row['checkpoint']}")

        if os.path.exists(model_path):
            models_to_evaluate = ["trained"]
            if random_transformer_baseline:
                models_to_evaluate.append("random")

            for model in models_to_evaluate:
                naming_suffix = f"-random" if model == "random" else ""
                # check if saved evaluation already exist
                if load_saved_evaluation and os.path.exists(os.path.join(model_path, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json")):
                    if add_to_df:
                        # load train set evaluation metrics and probs
                        transformer_df.at[index, (f'train_metrics{naming_suffix}')] = json.load(open(os.path.join(model_path, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json")))
                        transformer_df.at[index, (f'train_outputs{naming_suffix}')] = torch.load(os.path.join(model_path, f"outputs-train-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))
                        
                        # load eval set evaluation metrics and probs
                        transformer_df.at[index, (f'eval_metrics{naming_suffix}')] = json.load(open(os.path.join(model_path, f"metrics-eval-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json")))
                        transformer_df.at[index, (f'eval_outputs{naming_suffix}')] = torch.load(os.path.join(model_path, f"outputs-eval-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))

                        if exp_params["setting"] == "classification":
                            transformer_df.at[index, (f'iwl_metrics{naming_suffix}')] = json.load(open(os.path.join(model_path, f"metrics-iwl-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json")))
                            transformer_df.at[index, (f'iwl_outputs{naming_suffix}')] = torch.load(os.path.join(model_path, f"outputs-iwl-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))
                else:
                    # load model
                    if model == "trained":
                        model = load_transformer(config, os.path.join(model_path, "model"))
                    elif model == "random":
                        model = init_transformer(config)

                    # load datasets
                    train_dataset = init_dataset(config=config, mode="train", dataset_length=num_eval_sequences)
                    eval_dataset = init_dataset(config=config, mode="eval", dataset_length=num_eval_sequences)
                    
                    if exp_params["setting"] == "classification":
                        iwl_dataset = init_dataset(config=config, mode="train", dataset_length=num_eval_sequences, iwl=True)

                    compute_metrics = train_dataset.compute_metrics

                    # run evaluation
                    train_outputs, train_labels = collect_outputs_and_labels(model=model, dataset=train_dataset, batch_size=32, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)
                    eval_outputs, eval_labels = collect_outputs_and_labels(model=model, dataset=eval_dataset, batch_size=32, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)
                    
                    if exp_params["setting"] == "classification":
                        iwl_outputs, iwl_labels = collect_outputs_and_labels(model=model, dataset=iwl_dataset, batch_size=32, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)

                    # compute metrics
                    train_metrics = compute_metrics((train_outputs, train_labels), mode="train", return_all=True)
                    eval_metrics = compute_metrics((eval_outputs, eval_labels), mode="eval", return_all=True)
                    
                    if exp_params["setting"] == "classification":
                        iwl_metrics = compute_metrics((iwl_outputs, iwl_labels), mode="iwl", return_all=True)

                    # save metrics in model_path
                    with open(os.path.join(model_path, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json"), "w") as f:
                        json.dump(train_metrics, f)
                    with open(os.path.join(model_path, f"metrics-eval-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json"), "w") as f:
                        json.dump(eval_metrics, f)
                    
                    if exp_params["setting"] == "classification":
                        with open(os.path.join(model_path, f"metrics-iwl-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.json"), "w") as f:
                            json.dump(iwl_metrics, f)

                    # save outputs in model_path
                    torch.save(train_outputs.detach().cpu(), os.path.join(model_path, f"outputs-train-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))
                    torch.save(eval_outputs.detach().cpu(), os.path.join(model_path, f"outputs-eval-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))
                    
                    if exp_params["setting"] == "classification":
                        torch.save(iwl_outputs.detach().cpu(), os.path.join(model_path, f"outputs-iwl-{config.random_seed}seed-{num_eval_sequences}sequences{naming_suffix}.pt"))
                                    
                    if add_to_df:
                        # save metrics and probs in transformer_df
                        transformer_df.at[index, (f'train_outputs{naming_suffix}')] = train_outputs
                        transformer_df.at[index, (f'eval_outputs{naming_suffix}')] = eval_outputs
                        transformer_df.at[index, (f'train_metrics{naming_suffix}')] = train_metrics
                        transformer_df.at[index, (f'eval_metrics{naming_suffix}')] = eval_metrics
                        
                        if exp_params["setting"] == "classification":
                            transformer_df.at[index, (f'iwl_outputs{naming_suffix}')] = iwl_outputs
                            transformer_df.at[index, (f'iwl_metrics{naming_suffix}')] = iwl_metrics
        else:
            print(f"Model not found at {model_path}")
            pass

    return transformer_df
         
# Prediction with algorithms

def run_evaluation_algorithms(algo_df,
                            exp_params,
                            num_eval_sequences,
                            num_eval_sequences_nll_computation,
                            cache_dir,
                            load_saved_evaluation=True,
                            batch_size=16,
                            add_to_df=True, 
                            rng=None,
                            include_optimal_constant_solution=False):
    
    """Run full evaluation for algorithmic solutions.
    
    Returns algo_df with train and eval metrics and probs for both predictors.
    """
    algo_df = algo_df.copy() # make a copy to avoid modifying the original
    # create directory for algorithmic solutions
    algorithmic_solutions_dir = os.path.join(cache_dir, exp_params['setting'], "algorithmic-solutions")
    os.makedirs(algorithmic_solutions_dir, exist_ok=True)
    
    # Create empty columns for metrics and probs for both predictors
    config = algo_df.iloc[0]['config']
    algorithms = get_algorithmic_solutions(config, include_optimal_constant_solution=include_optimal_constant_solution)
    for algorithm_name, _ in algorithms.items():
        algo_df[f'{algorithm_name}_train_metrics'] = None
        algo_df[f'{algorithm_name}_eval_metrics'] = None
        algo_df[f'{algorithm_name}_train_outputs'] = None
        algo_df[f'{algorithm_name}_eval_outputs'] = None
        if exp_params["setting"] == "classification":
            algo_df[f'{algorithm_name}_iwl_metrics'] = None
            algo_df[f'{algorithm_name}_iwl_outputs'] = None

    # iterate over rows of algo_df

    for index, row in tqdm_func()(algo_df.iterrows(), total=algo_df.shape[0]):
        config = row['config']
        algorithms = get_algorithmic_solutions(config, include_optimal_constant_solution=include_optimal_constant_solution)

        # iterate over algorithms
        for algorithm_name, algorithm in algorithms.items():
            # make directory for algorithm
            algorithm_dir = os.path.join(algorithmic_solutions_dir, algorithm_name)
            os.makedirs(algorithm_dir, exist_ok=True)
            # directory for setting
            setting_dir = os.path.join(algorithm_dir, f"{config.num_dims}dims-{config.num_tasks}tasks-{config.context_length}context-{config.random_seed}seed"+("-"+config.name_suffix if config.name_suffix != "" else ""))

            # check if train and eval metrics and probs already exist
            if load_saved_evaluation and os.path.exists(os.path.join(setting_dir, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences.json")):
                if add_to_df:
                    # load train set evaluation metrics and probs
                    algo_df.at[index, (f'{algorithm_name}_train_metrics')] = json.load(open(os.path.join(setting_dir, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences.json")))
                    algo_df.at[index, (f'{algorithm_name}_train_outputs')] = torch.load(os.path.join(setting_dir, f"outputs-train-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))

                    # load robust nll estimate
                    algo_df.at[index, (f'{algorithm_name}_id_nll')] = json.load(open(os.path.join(setting_dir, f"id_nll_MOM-{config.random_seed}seed-{num_eval_sequences_nll_computation}sequences.json")))
                    
                    # load eval set evaluation metrics and probs
                    algo_df.at[index, (f'{algorithm_name}_eval_metrics')] = json.load(open(os.path.join(setting_dir, f"metrics-eval-{config.random_seed}seed-{num_eval_sequences}sequences.json")))
                    algo_df.at[index, (f'{algorithm_name}_eval_outputs')] = torch.load(os.path.join(setting_dir, f"outputs-eval-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))

                    if exp_params["setting"] == "classification":
                        algo_df.at[index, (f'{algorithm_name}_iwl_metrics')] = json.load(open(os.path.join(setting_dir, f"metrics-iwl-{config.random_seed}seed-{num_eval_sequences}sequences.json")))
                        algo_df.at[index, (f'{algorithm_name}_iwl_outputs')] = torch.load(os.path.join(setting_dir, f"outputs-iwl-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))
            else:
                # make directory for setting
                os.makedirs(setting_dir, exist_ok=True)

                # save yaml in setting_dir
                yaml_str = config.make_yaml_str()
                with open(os.path.join(setting_dir, "config.yaml"), "w") as f:
                    f.write(yaml_str)

                # get train and eval data
                train_data = init_dataset(config, "train", dataset_length=num_eval_sequences)
                eval_data = init_dataset(config, "eval", dataset_length=num_eval_sequences)

                if exp_params["setting"] == "classification":
                    iwl_data = init_dataset(config, "train", dataset_length=num_eval_sequences, iwl=True)

                compute_metrics = train_data.compute_metrics

                # run train and eval
                train_outputs, train_labels = collect_outputs_and_labels(model=algorithm, dataset=train_data, batch_size=batch_size, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)
                eval_outputs, eval_labels = collect_outputs_and_labels(model=algorithm, dataset=eval_data, batch_size=batch_size, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)
                
                if exp_params["setting"] == "classification":
                    iwl_outputs, iwl_labels = collect_outputs_and_labels(model=algorithm, dataset=iwl_data, batch_size=batch_size, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)

                # check if config has an attribute "noise_variance"
                if hasattr(config, "noise_variance"):
                    noise_variance = config.noise_variance
                    train_metrics = compute_metrics((train_outputs, train_labels), mode="train", return_all=True, noise_variance=noise_variance, compute_nll=True)
                    eval_metrics = compute_metrics((eval_outputs, eval_labels), mode="eval", return_all=True, noise_variance=noise_variance, compute_nll=True)
                else:
                    train_metrics = compute_metrics((train_outputs, train_labels), mode="train", return_all=True, compute_nll=True)
                    eval_metrics = compute_metrics((eval_outputs, eval_labels), mode="eval", return_all=True, compute_nll=True)

                if exp_params["setting"] == "classification":
                    iwl_metrics = compute_metrics((iwl_outputs, iwl_labels), mode="iwl", return_all=True, compute_nll=True)

                # save metrics in model_path
                with open(os.path.join(setting_dir, f"metrics-train-{config.random_seed}seed-{num_eval_sequences}sequences.json"), "w") as f:
                    json.dump(train_metrics, f)
                with open(os.path.join(setting_dir, f"metrics-eval-{config.random_seed}seed-{num_eval_sequences}sequences.json"), "w") as f:
                    json.dump(eval_metrics, f)
                
                if exp_params["setting"] == "classification":
                    with open(os.path.join(setting_dir, f"metrics-iwl-{config.random_seed}seed-{num_eval_sequences}sequences.json"), "w") as f:
                        json.dump(iwl_metrics, f)

                # save probs in model_path
                torch.save(train_outputs.detach().cpu(), os.path.join(setting_dir, f"outputs-train-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))
                torch.save(eval_outputs.detach().cpu(), os.path.join(setting_dir, f"outputs-eval-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))
                
                if exp_params["setting"] == "classification":
                    torch.save(iwl_outputs.detach().cpu(), os.path.join(setting_dir, f"outputs-iwl-{config.random_seed}seed-{num_eval_sequences}sequences.pt"))

                if add_to_df:
                    # save metrics and probs in algo_df
                    algo_df.at[index, (f'{algorithm_name}_train_metrics')] = train_metrics
                    algo_df.at[index, (f'{algorithm_name}_eval_metrics')] = eval_metrics
                    algo_df.at[index, (f'{algorithm_name}_train_outputs')] = train_outputs
                    algo_df.at[index, (f'{algorithm_name}_eval_outputs')] = eval_outputs
                    
                    if exp_params["setting"] == "classification":
                        algo_df.at[index, (f'{algorithm_name}_iwl_metrics')] = iwl_metrics
                        algo_df.at[index, (f'{algorithm_name}_iwl_outputs')] = iwl_outputs

                # compute nll via median of means for increased robustness to outliers
                ## get train data
                train_data_nll_computation = init_dataset(config, "train", dataset_length=num_eval_sequences_nll_computation)
                train_outputs_nll_computation, train_labels_nll_computation = collect_outputs_and_labels(model=algorithm, dataset=train_data_nll_computation, batch_size=batch_size, labels_key=config.labels_key, inputs_key=config.inputs_key, collect_all_batches=True)

                # check if config has an attribute "noise_variance"
                if hasattr(config, "noise_variance"):
                    noise_variance = config.noise_variance
                    train_metrics_nll_computation = compute_metrics((train_outputs_nll_computation, train_labels_nll_computation), mode="train", return_all=True, noise_variance=noise_variance, compute_nll=True)
                else:
                    train_metrics_nll_computation = compute_metrics((train_outputs_nll_computation, train_labels_nll_computation), mode="train", return_all=True, compute_nll=True)

                # compute nll via median of means
                train_nll_mom = median_of_means(np.array(train_metrics_nll_computation['train_all_nll']), rng=rng)

                # save estimate
                with open(os.path.join(setting_dir, f"id_nll_MOM-{config.random_seed}seed-{num_eval_sequences_nll_computation}sequences.json"), "w") as f:
                    json.dump(train_nll_mom, f)

                if add_to_df:
                    algo_df.at[index, (f'{algorithm_name}_id_nll')] = train_nll_mom

    return algo_df


def find_approximate_interpolation_threshold(df, threshold_percentile=0.2):
    """
    Find the approximate interpolation threshold for each configuration group.
    
    Args:
        df (pd.DataFrame): DataFrame containing the transformer data
        threshold_percentile (float): Percentile value for interpolation_distance_train
        hard_threshold (float): Hard threshold for interpolation_distance_train
        
    Returns:
        pd.DataFrame: DataFrame with added columns for interpolation threshold
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # First, find the threshold for each (mlp, dims, context, num_tasks) group
    group_thresholds = {}
    
    # Group by all parameters and find first checkpoint below threshold
    for (mlp, dims, context, tasks), group in df.groupby(['mlp_expansion_factor', 'num_dims', 'context_length', 'num_tasks']):
        # Sort by checkpoint to ensure we find the first occurrence
        sorted_group = group.sort_values('checkpoint')

        # find max and min values of interpolation_distance_train
        max_value = sorted_group['interpolation_distance_train'].max()
        min_value = sorted_group['interpolation_distance_train'].min()
        threshold = min_value + (max_value - min_value) * threshold_percentile # determine threshold as a percentile of the range

        # Find first checkpoint where interpolation_distance_train is below threshold
        below_threshold = sorted_group.query("interpolation_distance_train < @threshold", engine="python")
        
        if not below_threshold.empty:
            first_checkpoint = below_threshold['checkpoint'].iloc[0]
            group_thresholds[(mlp, dims, context, tasks)] = first_checkpoint
        else:
            # If no checkpoint is below threshold, return inf
            group_thresholds[(mlp, dims, context, tasks)] = np.inf
    
    # For each (mlp, dims, context) group, find the maximum threshold across all num_tasks
    config_max_thresholds = {}
    for (mlp, dims, context, tasks), threshold_value in group_thresholds.items():
        config_key = (mlp, dims, context)
        if config_key in config_max_thresholds:
            config_max_thresholds[config_key] = max(config_max_thresholds[config_key], threshold_value)
        else:
            config_max_thresholds[config_key] = threshold_value
    
    # Add the threshold to each row and create the inclusion column
    result_df['approximate_interpolation_threshold'] = 0
    result_df['included_in_interpolation_analysis'] = 0
    
    for idx, row in result_df.iterrows():
        config_key = (row['mlp_expansion_factor'], row['num_dims'], row['context_length'])
        if config_key in config_max_thresholds:
            threshold_value = config_max_thresholds[config_key]
            result_df.at[idx, 'approximate_interpolation_threshold'] = threshold_value
            result_df.at[idx, 'included_in_interpolation_analysis'] = 1 if row['checkpoint'] >= threshold_value else 0
    
    return result_df

##########################
# Distance and barycentric helper functions
##########################

def compute_barycentric_weights(transformer_out, algo_outs, distance_function):
    """
    Compute barycentric coordinates for transformer output relative to N algorithmic predictors.
    For N=2, use the original formula. For N>2, fit the optimal convex combination (LIA weights)
    using torch.optim (LBFGS) directly, with softmax weights.
    
    Args:
        transformer_out: Transformer output tensor
        algo_outs: Dict of algorithm outputs {algo_name: tensor}
        distance_function: Distance function to use (function or string)
    
    Returns:
        weights: Dict of barycentric weights {algo_name: weight}
        interpolated_out: Weighted combination using these weights
        interpolation_distance: Distance from transformer to interpolated point
        individual_distances: Dict of distances from transformer to each algorithm
    """
    import torch.optim as optim
    import torch.nn as nn

    if distance_function == "MSE":
        distance_fn = lambda x, y: torch.nn.MSELoss()(x, y) / x.shape[-1]  # Normalize by dimension
    elif distance_function == "KL":
        # apply symmetrized KL divergence instead of standard KL divergence
        distance_fn = lambda p_t, p_algo: (torch.nn.KLDivLoss(reduction='none')(p_t.log(), p_algo).sum(dim=-1).mean() \
                                                + torch.nn.KLDivLoss(reduction='none')(p_algo.log(), p_t).sum(dim=-1).mean()) / 2
    else:
        raise ValueError(f"Distance function {distance_function} not supported")

    algo_names = list(algo_outs.keys())
    N = len(algo_names)

    # Compute individual distances first
    individual_distances = {}
    for algo_name in algo_names:
        individual_distances[algo_name] = distance_fn(transformer_out, algo_outs[algo_name]).item()

    if N == 2:
        # Use original formula for N=2 case (maintains exact backward compatibility)
        A_name, B_name = algo_names[0], algo_names[1]
        A, B = algo_outs[A_name], algo_outs[B_name]

        d_TA = individual_distances[A_name]
        d_TB = individual_distances[B_name]
        d_AB = distance_fn(A, B).item()

        r = (d_TB - d_TA) / d_AB
        r = torch.clamp(torch.tensor(r), -1, 1).item()  # Keep original clamping
        w_A = (r + 1) / 2
        w_B = 1 - w_A

        weights = {A_name: w_A, B_name: w_B}

        # Compute interpolated output
        interpolated_out = w_A * A + w_B * B
        interpolation_distance = distance_fn(transformer_out, interpolated_out).item()

    else:
        # N > 2: Fit optimal convex combination (LIA weights) using torch.optim
        algo_stack = torch.stack([algo_outs[name] for name in algo_names], dim=-1)  # (..., N)
        # flatten all but last dimension
        device = algo_stack.device
        dtype = algo_stack.dtype

        # Initialize weights uniformly
        lia_weights = torch.ones(N, device=device, dtype=dtype).requires_grad_()

        optimizer = optim.LBFGS([lia_weights], max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            w = torch.softmax(lia_weights, dim=0)
            interpolated = (algo_stack * w).sum(dim=-1)
            loss = distance_fn(transformer_out, interpolated)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            w = torch.softmax(lia_weights, dim=0)
            interpolated_out = (algo_stack * w).sum(dim=-1)
            interpolation_distance = distance_fn(transformer_out, interpolated_out).item()
            weights = {name: w[i].item() for i, name in enumerate(algo_names)}

    return weights, interpolation_distance, individual_distances

##########################
# main analysis
##########################
    
def compute_distances(transformer_df, algo_df, remove_last_prediction=False, distance_function="KL", include_optimal_constant_solution=False):
    """Compute distances between transformer and algorithm outputs (generalized to N predictors)
    """
    # get cuda if available
    device = get_device()

    # Get list of algorithms
    algos = list(algo_df.iloc[0]['config'].algo_names_dict.keys())
    if include_optimal_constant_solution:
        algos.append("optimal_constant")
    # Add columns for individual distances from each algorithm (backward compatibility)
    for algo in algos:
        # add columns for individual distances from each algorithm (backward compatibility)
        transformer_df[f"distance_from_{algo}_train"] = None
        transformer_df[f"distance_from_{algo}_eval"] = None
        # add columns for barycentric weights
        transformer_df[f"barycentric_weight_{algo}_train"] = None
        transformer_df[f"barycentric_weight_{algo}_eval"] = None
    
    # Add interpolation distance column
    transformer_df["interpolation_distance_train"] = None
    transformer_df["interpolation_distance_eval"] = None
    
    if len(algos) == 2:
        transformer_df["relative_distance_train"] = None
        transformer_df["relative_distance_eval"] = None
    else:
        transformer_df["barycentric_weights_train"] = None
        transformer_df["barycentric_weights_eval"] = None
    
    
    for idx, row in tqdm_func()(transformer_df.iterrows(), total=len(transformer_df)):
        
        matching = algo_df.query(
            "num_tasks == @row.num_tasks & num_dims == @row.num_dims & context_length == @row.context_length", engine="python"
        )
        if matching.empty:
            continue
        a_row = matching.iloc[0]

        # get metric_name from config
        metric_name = row['config'].metric_name

        for eval_mode in ["train", "eval"]:
            # Prepare algorithm outputs
            algo_outs = {}
            for algo in algos:
                if metric_name == "MSE":
                    algo_outs[algo] = torch.as_tensor(a_row[f"{algo}_{eval_mode}_outputs"], device=device, dtype=torch.float)
                else:
                    algo_outs[algo] = torch.softmax(torch.as_tensor(a_row[f"{algo}_{eval_mode}_outputs"], device=device, dtype=torch.float), dim=-1)
            
            # Prepare transformer output
            if metric_name == "MSE":
                transformer_outs = torch.as_tensor(row[f"{eval_mode}_outputs"], device=device, dtype=torch.float)
            else:
                transformer_outs = torch.softmax(torch.as_tensor(row[f"{eval_mode}_outputs"], device=device, dtype=torch.float), dim=-1)
            
            if remove_last_prediction:
                algo_outs = {k: v[..., :-1, :] for k, v in algo_outs.items()}
                transformer_outs = transformer_outs[..., :-1, :]

            # Compute barycentric weights and distances
            weights, interp_distance, individual_distances = compute_barycentric_weights(
                transformer_outs, algo_outs, distance_function
            )
            
            # Store barycentric weights
            for algo in algos:
                transformer_df.at[idx, f"barycentric_weight_{algo}_{eval_mode}"] = weights[algo]
            
            # Store individual distances
            for algo in algos:
                transformer_df.at[idx, f"distance_from_{algo}_{eval_mode}"] = individual_distances[algo]
            
            # Store interpolation distance
            if metric_name == "MSE":
                transformer_df.at[idx, f"interpolation_distance_{eval_mode}"] = interp_distance / row["config"].num_dims
            else:
                transformer_df.at[idx, f"interpolation_distance_{eval_mode}"] = interp_distance
            
            if len(algos) == 2:
                first_algo = algos[0]
                transformer_df.at[idx, f"relative_distance_{eval_mode}"] = weights[first_algo]

            else:
                transformer_df.at[idx, f"barycentric_weights_{eval_mode}"] = np.array(list(weights.values()))

    return transformer_df

##########################
# main analysis
##########################
    
def compute_distances(transformer_df, algo_df, remove_last_prediction=False, distance_function="KL", include_optimal_constant_solution=False):
    """Compute distances between transformer and algorithm outputs (generalized to N predictors)
    """
    # get cuda if available
    device = get_device()

    # Get list of algorithms
    algos = list(algo_df.iloc[0]['config'].algo_names_dict.keys())
    if include_optimal_constant_solution:
        algos.append("optimal_constant")
    # Add columns for individual distances from each algorithm (backward compatibility)
    for algo in algos:
        # add columns for individual distances from each algorithm (backward compatibility)
        transformer_df[f"distance_from_{algo}_train"] = None
        transformer_df[f"distance_from_{algo}_eval"] = None
        # add columns for barycentric weights
        transformer_df[f"barycentric_weight_{algo}_train"] = None
        transformer_df[f"barycentric_weight_{algo}_eval"] = None
    
    # Add interpolation distance column
    transformer_df["interpolation_distance_train"] = None
    transformer_df["interpolation_distance_eval"] = None
    
    if len(algos) == 2:
        transformer_df["relative_distance_train"] = None
        transformer_df["relative_distance_eval"] = None
    else:
        transformer_df["barycentric_weights_train"] = None
        transformer_df["barycentric_weights_eval"] = None
    
    
    for idx, row in tqdm_func()(transformer_df.iterrows(), total=len(transformer_df)):
        # Debug output for specific case
        
        matching = algo_df.query(
            "num_tasks == @row.num_tasks & num_dims == @row.num_dims & context_length == @row.context_length", engine="python"
        )
        if matching.empty:
            continue
        a_row = matching.iloc[0]

        # get metric_name from config
        metric_name = row['config'].metric_name

        for eval_mode in ["train", "eval"]:
            # Prepare algorithm outputs
            algo_outs = {}
            for algo in algos:
                if metric_name == "MSE":
                    algo_outs[algo] = torch.as_tensor(a_row[f"{algo}_{eval_mode}_outputs"], device=device, dtype=torch.float)
                else:
                    algo_outs[algo] = torch.softmax(torch.as_tensor(a_row[f"{algo}_{eval_mode}_outputs"], device=device, dtype=torch.float), dim=-1)
            
            # Prepare transformer output
            if metric_name == "MSE":
                transformer_outs = torch.as_tensor(row[f"{eval_mode}_outputs"], device=device, dtype=torch.float)
            else:
                transformer_outs = torch.softmax(torch.as_tensor(row[f"{eval_mode}_outputs"], device=device, dtype=torch.float), dim=-1)
            
            if remove_last_prediction:
                algo_outs = {k: v[..., :-1, :] for k, v in algo_outs.items()}
                transformer_outs = transformer_outs[..., :-1, :]

            # Compute barycentric weights and distances
            weights, interp_distance, individual_distances = compute_barycentric_weights(
                transformer_outs, algo_outs, distance_function
            )
            
            # Store barycentric weights
            for algo in algos:
                transformer_df.at[idx, f"barycentric_weight_{algo}_{eval_mode}"] = weights[algo]
            
            # Store individual distances
            for algo in algos:
                transformer_df.at[idx, f"distance_from_{algo}_{eval_mode}"] = individual_distances[algo]
            
            # Store interpolation distance
            if metric_name == "MSE":
                transformer_df.at[idx, f"interpolation_distance_{eval_mode}"] = interp_distance / row["config"].num_dims
            else:
                transformer_df.at[idx, f"interpolation_distance_{eval_mode}"] = interp_distance
            
            if len(algos) == 2:
                first_algo = algos[0]
                transformer_df.at[idx, f"relative_distance_{eval_mode}"] = weights[first_algo]

            else:
                transformer_df.at[idx, f"barycentric_weights_{eval_mode}"] = np.array(list(weights.values()))

    return transformer_df


def compute_derivatives(df, column_name, group_by_vars, vary_by_var,
                        use_log_scale=True, result_suffix=None, method="logistic_fit"):
    """
    Compute first and second derivatives of a specified column with respect to a varying variable,
    while holding other variables constant. First smooths the data, then calculates derivatives.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to compute derivatives for
        group_by_vars: List of column names to group by (variables to hold constant)
        vary_by_var: Column name of the variable to compute derivatives with respect to
        use_log_scale: Whether to use log scale for the variable (True for power-law relationships)
        result_suffix: Suffix for the output columns (defaults to vary_by_var if None)
    
    Returns:
        DataFrame with first and second derivatives added
    """    
    # Set suffix for result columns
    if result_suffix is None:
        result_suffix = vary_by_var
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add columns for derivatives
    first_derivative_col = f"{column_name}_first_derivative_over_{result_suffix}"
    second_derivative_col = f"{column_name}_second_derivative_over_{result_suffix}"
    result_df[first_derivative_col] = np.nan
    result_df[second_derivative_col] = np.nan

    if method == "logistic_fit":
        logistic = lambda x, x0, k, l: l / (1 + np.exp(-k * (x - x0)))
        dlogistic = lambda x, x0, k, l: k*logistic(x, x0, k, l)*(1-logistic(x, x0, k, l)/l)
        ddlogistic = lambda x, x0, k, l: k**2*logistic(x, x0, k, l)*(1-logistic(x, x0, k, l)/l)*(1-2*logistic(x, x0, k, l)/l)
        # set up logistic fit param col
        result_df[f"{column_name}_logistic_fit"] = None
    
    # Compute derivatives for each group
    for name, group in result_df.groupby(group_by_vars):

        # Sort by the varying variable
        sorted_group = group.sort_values(vary_by_var)
        
        # Get the indices for later mapping back
        original_indices = sorted_group.index.tolist()
        
        # Get values to compute derivatives
        x_values = np.array(sorted_group[vary_by_var].values)
        y_values = np.array(sorted_group[column_name].values)
        
        # Apply log transformation if requested
        if use_log_scale:
            x_values = np.log(x_values)

        if method == "logistic_fit":
            # initial guess for x0 and k
            p0 = [np.median(x_values), 0.5, np.median(y_values)]
            # fit logistic function
            popt, _ = curve_fit(logistic, x_values, y_values, p0=p0, maxfev=100000)
            x0, k, l = popt
            first_derivative = dlogistic(x_values, x0, k, l)
            second_derivative = ddlogistic(x_values, x0, k, l)
            # compute goodness of fit with r2 score
            r2 = r2_score(y_values, logistic(x_values, x0, k, l))
            # set column values for group
            result_df.loc[original_indices, f"{column_name}_logistic_fit"] = {"x0": x0, "k": k, "l": l, "r2": r2}
        else:
            # apply numpy gradient to get first and second derivatives with method of differences
            first_derivative = np.gradient(y_values, x_values)
            second_derivative = np.gradient(first_derivative, x_values)

        # map back to original indices
        result_df.loc[original_indices, first_derivative_col] = first_derivative
        result_df.loc[original_indices, second_derivative_col] = second_derivative
    
    return result_df


def get_algorithm_complexities(algo_df, increase_generalized_code_complexity=False, include_optimal_constant_solution=False):
    """compute the upper bound on algorithmic complexity for predictors
    """
    algo_complexity_dict = {}
    algos = get_algorithmic_solutions(algo_df.iloc[0]["config"], include_optimal_constant_solution=include_optimal_constant_solution)
    # code complexity doesn't depend on any variables, so we can just compute it once
    for algo in algos:
        algo_df[f"{algo}_complexity"] = None
        algorithm_class = algos[algo].__class__.__name__
        code_complexity = compress_python_script(algorithm_class)
        algo_complexity_dict[algo] = {}
        if algo == "generalized" and increase_generalized_code_complexity:
            code_complexity = code_complexity*5
        algo_complexity_dict[algo]["code_complexity"] = code_complexity
    
    # iterate over rows in algo_df
    for idx, row in tqdm_func()(algo_df.iterrows(), total=len(algo_df)):
        algos = get_algorithmic_solutions(row["config"], include_optimal_constant_solution=include_optimal_constant_solution)
        num_dims = row["num_dims"]
        num_tasks = row["num_tasks"]
        for algo in algos:
            if (num_tasks, num_dims) not in algo_complexity_dict[algo]:
                algo_prior = algos[algo].prior
                algo_complexity_dict[algo][(num_tasks, num_dims)] = compress_array(np.array(algo_prior))
            algo_df.at[idx, f"{algo}_complexity"] = algo_complexity_dict[algo][(num_tasks, num_dims)] + algo_complexity_dict[algo]["code_complexity"]
    return algo_df

class ScipyOptimizer:
    """Optimize model parameters using scipy and gradients from torch.
    """

    @classmethod
    def train_model(cls, model, batch_list, 
                    method="L-BFGS-B",
                    ftol=1e-7, gtol=1e-7, return_history=True):
        """
        Trains a model by optimizing scalar parameters using scipy.minimize.
        
        Args:
            model (nn.Module): The instance of the model to train.
            batch_list (list): List of batches, where each batch is a dictionary with keys
                            needed for the model's forward pass.
            initial_param_search (bool): Whether run initial parameter search.
            ranges (list): List of tuples, where each tuple contains the range of values to search over for one parameter.
            method (str): The method to use for optimization.
            maxiter_initial_search (int): The maximum number of iterations for the initial parameter search.
            return_history (bool): Whether to return the history of the optimization.
        """
        # Move model to device
        device = get_device()
        model.to(device)
        # Move all batches to device once
        processed_batches = [move_batch_to_device(batch, device) for batch in batch_list]
        # Get all scalar parameters that require gradients
        parameters = [p for p in model.parameters() if p.requires_grad and p.numel() == 1]


        # set model, device, and processed batches attributes
        cls.model = model
        cls.device = device
        cls.processed_batches = processed_batches
        cls.parameters = parameters
                
        # Get initial parameter values
        initial_params = np.array([p.item() for p in parameters])
        
        # Print initial parameters info
        print(f"Starting optimization with {len(initial_params)} parameters")
        print(f"Initial parameter values: {initial_params}")


        # then use gradient based optimizer to fine-tune
        if method == "L-BFGS-B":
            cls.use_grad = True
            print("Optimizing with L-BFGS-B...")
            result = optimize.minimize(
                    cls.objective_function,
                    initial_params,
                    method=method,
                    options={
                        'disp': True, 
                        'maxiter': 1000,
                        'maxfun': 2000,      
                        'gtol': gtol,       
                        'ftol': ftol      
                    },
                    jac=cls.use_grad
                )
        elif method == "Nelder-Mead":
            cls.use_grad = False
            print("Optimizing with Nelder-Mead...")
            result = optimize.minimize(
                cls.objective_function,
                initial_params,
                method='Nelder-Mead',
                options={
                'disp': True,         # Show progress
                'maxiter': 200,       # Reasonable upper limit on iterations
                'maxfev': 50,    
                'xatol': 1e-5,        # Convergence tolerance on parameters
                'fatol': 1e-5,        # Convergence tolerance on objective value
                'return_all': False   # Set True if you want the full solution path
            }
            )
        else:
            raise ValueError(f"Method {method} not supported")

        
        # Update the model with optimal parameters
        for i, param in enumerate(cls.parameters):
            param.data = torch.tensor(result.x[i], device=cls.device)
        
        # Print optimization results
        print(f"Optimization complete with final loss: {result.fun:.6f}")
        print(f"Final parameter values: {result.x} (when exponentiated: {np.exp(result.x)})")

        if return_history:
            return {
                'optimization_result': {
                    'success': result.success,
                    'status': result.status,
                    'message': result.message,
                    'fun': result.fun,
                    'nit': result.nit,
                    'nfev': result.nfev,
                },
                'final_params': result.x,
            }

        
    @classmethod
    def objective_function(cls, params):
        # update model parameters
        for i, param in enumerate(cls.parameters):
            param.data = torch.tensor(params[i], device=cls.device, requires_grad=True)
        
        # Compute average loss over all batches
        total_loss = 0.0
        for batch in tqdm(cls.processed_batches, desc="Evaluating batches", leave=False):
            # Forward pass with the current parameter values
            outputs = cls.model(**batch)
            total_loss += outputs["loss"]
        
        mean_loss = total_loss / len(cls.processed_batches)
        
        # Compute gradients if needed
        if cls.use_grad:
            # previous steps to accumulate
            if hasattr(cls.model, 'zero_grad'):
                cls.model.zero_grad()
                
            mean_loss.backward()
            
            # Collect gradients
            grads = np.array([param.grad.item() for param in cls.parameters])
            return mean_loss.item(), grads
        else:
            return mean_loss.item()

class HierarchicalBayesianModelFitter:
    """
    A class to fit Hierarchical Bayesian Model with optimized batched operations.
    
    This class separates the complex fitting process into logical components:
    - Data preparation and caching
    - Model creation and training
    - Evaluation and result saving
    """
    
    def __init__(self, transformer_df, algo_df, mlp_expansion_factor, context_length, 
                 num_dims, metric_name, params_init, **kwargs):
        """
        Initialize the fitter with configuration parameters.
        
        Args:
            transformer_df: DataFrame containing transformer results
            algo_df: DataFrame containing algorithm results
            mlp_expansion_factor: Only fit data with this MLP expansion factor
            context_length: Context length to use
            num_dims: Number of dimensions
            metric_name: Metric name to use
            params_init: Initial parameters for the model
            **kwargs: Additional configuration options
        """
        # Store core parameters
        self.transformer_df = transformer_df
        self.algo_df = algo_df
        self.mlp_expansion_factor = mlp_expansion_factor
        self.context_length = context_length
        self.num_dims = num_dims
        self.metric_name = metric_name
        self.params_init = params_init
        self.batch_size = transformer_df["config"].iloc[0].per_device_train_batch_size
        print("batch_size: ", self.batch_size)
        
        # Store configuration options with defaults
        self.load_saved_evaluation = kwargs.get('load_saved_evaluation', True)
        self.add_to_df = kwargs.get('add_to_df', True)
        self.remove_last_prediction = kwargs.get('remove_last_prediction', True)
        self.out_column = kwargs.get('out_column', "bms_results")
        self.fit_params = kwargs.get('fit_params', True)
        self.fit_with_ood = kwargs.get('fit_with_ood', False)
        self.fixed_params = kwargs.get('fixed_params', [])
        self.percent_train = kwargs.get('percent_train', 0.8)
        self.method = kwargs.get('method', "L-BFGS-B")
        self.context_dependent_weights = kwargs.get('context_dependent_weights', False)
        self.ftol = kwargs.get('ftol', 1e-7)
        self.gtol = kwargs.get('gtol', 1e-7)
        self.baseline_lst = kwargs.get('baseline_lst', [])
        self.return_history = kwargs.get('return_history', False)
        self.ablation_model = kwargs.get('ablation_model', False)
        self.ablation_model_args = kwargs.get('ablation_model_args', {})
        self.include_optimal_constant_solution = kwargs.get('include_optimal_constant_solution', False)
        
        # Initialize state variables
        self.filtered_df = None
        self.merged_df = None
        self.train_checkpoints = None
        self.train_num_tasks = None
        self.loss_type = None
        self.algos = None
        self.device = torch.device("cpu")
        self.model = None
        self.history = []
        
        # Data storage
        self.transformer_outs_id = None
        self.transformer_outs_ood = None
        self.algo_outs_id = None
        self.algo_outs_ood = None
        self.algo_losses_id = None
        self.algo_losses_ood = None
        
        # Performance optimization: Pre-computed lookup tables
        self.complexity_lookup = {}
        self.avg_losses_lookup = {}
        
        # Baseline computer (initialized after data preparation)
        self.baseline_computer = None
    
    def fit(self):
        """
        Main fitting method that orchestrates the entire process.
        
        Returns:
            tuple: (transformer_df, history, model)
        """
        # Setup initial state
        self._setup_initial_state()
        
        # Try to load saved results
        if self._load_saved_results():
            return self.transformer_df, self.history, self.model
        
        # Prepare data
        self._prepare_data()
        
        # Create and train model
        self._create_model()
        self._train_model()
        
        # Evaluate and save results
        self._evaluate_and_save()
        
        # Write correlation results
        if not self.ablation_model and self.fit_params and not self.context_dependent_weights:
            self._write_average_evaluation_results()
        
        # Clean up memory
        gc.collect()
        
        return self.transformer_df, self.history, self.model
    
    def _setup_initial_state(self):
        """Setup initial state and validate parameters."""
        # Setup output column
        if self.add_to_df and self.out_column not in self.transformer_df.columns:
            self.transformer_df[self.out_column] = None
        
        # Filter rows by constant values
        self.filtered_df = self.transformer_df[
            (self.transformer_df["mlp_expansion_factor"] == self.mlp_expansion_factor) &
            (self.transformer_df["context_length"] == self.context_length) &
            (self.transformer_df["num_dims"] == self.num_dims)
        ].copy()
        
        if self.filtered_df.empty:
            raise ValueError("No matching checkpoints or MLP expansion factor found.")
        
        # Setup training/test split
        checkpoints = self.filtered_df["checkpoint"].unique()
        num_tasks = self.filtered_df["num_tasks"].unique()
        
        self.train_checkpoints = checkpoints[:ceil(len(checkpoints) * self.percent_train)]
        self.train_num_tasks = num_tasks[:ceil(len(num_tasks) * self.percent_train)]
        
        print("train_checkpoints:", self.train_checkpoints, "\ntrain_num_tasks:", self.train_num_tasks)
        
        # Setup algorithm and loss type info
        self.setting = self.filtered_df["config"].iloc[0].setting
        self.loss_type = "MSE" if self.filtered_df["config"].iloc[0].metric_name == "MSE" else "KL"
        self.algos = list(self.filtered_df["config"].iloc[0].algo_names_dict.keys())
        if self.include_optimal_constant_solution:
            self.algos = self.algos.copy() + ["optimal_constant"]
        
        # Pre-compute lookup tables for performance optimization
        self.complexity_lookup, self.avg_losses_lookup = self.create_posterior_computation_lookup_tables()

    
    def create_posterior_computation_lookup_tables(self):
        """
        Create lookup tables for HierarchicalBayesianModel to avoid pandas queries during evaluation.
        
        Args:
            algo_df (pandas.DataFrame): DataFrame containing algorithm results
            algos (list): List of algorithm names to include in lookup tables
            
        Returns:
            tuple: (complexity_lookup, avg_losses_lookup)
        """
        complexity_lookup = {}
        avg_losses_lookup = {}
        
        # Get unique combinations of experimental parameters
        unique_combinations = self.algo_df[['num_tasks', 'num_dims', 'context_length']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            num_tasks, num_dims, context_length = row['num_tasks'], row['num_dims'], row['context_length']
            key = (num_tasks, num_dims, context_length)
            
            # Query algo_df once and cache results
            algo_row = self.algo_df.query(
                "num_tasks == @num_tasks & num_dims == @num_dims & context_length == @context_length", engine="python"
            ).iloc[0]
            
            # Pre-compute complexity values
            complexity_values = []
            for algo in self.algos:
                if f'{algo}_complexity' in algo_row:
                    complexity_values.append(algo_row[f'{algo}_complexity'])
            complexity_lookup[key] = torch.tensor(complexity_values, dtype=torch.float)
            
            # Pre-compute average losses
            avg_losses = []
            for algo in self.algos:
                if f'{algo}_id_nll' in algo_row:
                    avg_losses.append(algo_row[f'{algo}_id_nll'])
            avg_losses_lookup[key] = torch.tensor(avg_losses, dtype=torch.float)
        
        return complexity_lookup, avg_losses_lookup

    def _load_saved_results(self):
        """Load saved results if they exist and are complete."""
        if not self.load_saved_evaluation:
            return False
        
        all_results_exist = all(
            os.path.exists(os.path.join(row["config"].output_dir, f"checkpoint-{row['checkpoint']}", f"{self.out_column}.json"))
            for _, row in self.filtered_df.iterrows()
        )
        
        if all_results_exist:
            for idx, row in tqdm_func()(self.filtered_df.iterrows(), total=len(self.filtered_df), desc="Loading saved BMS results for all rows"):
                bms_results_file = os.path.join(row["config"].output_dir, f"checkpoint-{row['checkpoint']}", f"{self.out_column}.json")
                with open(bms_results_file, "r") as f:
                    if self.add_to_df:
                        self.transformer_df.at[idx, self.out_column] = json.load(f)
            
            # Instantiate model from fitted params
            row_with_fitted_params = self.transformer_df[~self.transformer_df[self.out_column].isna()].iloc[0]
            log_alpha = row_with_fitted_params[self.out_column]["params"]["log_alpha"]
            log_beta = row_with_fitted_params[self.out_column]["params"]["log_beta"]
            log_gamma = row_with_fitted_params[self.out_column]["params"]["log_gamma"]
            self.model = HierarchicalBayesianModel(
                params_init={"log_alpha": log_alpha, "log_beta": log_beta, "log_gamma": log_gamma}, 
                complexity_lookup=self.complexity_lookup,
                avg_losses_lookup=self.avg_losses_lookup,
                metric_name=self.metric_name
            )
            return True
        
        return False
    
    def _prepare_data(self):
        """Prepare and merge dataframes, then create tensors."""
        # Create merge keys for efficient joining
        self.filtered_df['merge_key'] = self.filtered_df.apply(
            lambda row: f"{row['num_tasks']}_{row['num_dims']}_{row['context_length']}", axis=1
        )
        self.algo_df['merge_key'] = self.algo_df.apply(
            lambda row: f"{row['num_tasks']}_{row['num_dims']}_{row['context_length']}", axis=1
        )
        
        # Merge dataframes once instead of repeated queries
        self.merged_df = self.filtered_df.merge(self.algo_df, on='merge_key', how='left', suffixes=('', '_algo'))
        
        if self.merged_df.empty:
            raise ValueError("No matching algorithm data found.")
        
        print(f"Processing {len(self.merged_df)} rows...")
        
        # Prepare tensors for all rows
        self.transformer_outs_id, self.algo_outs_id, self.algo_losses_id = self._get_transformer_and_algo_outs("train")
        self.transformer_outs_ood, self.algo_outs_ood, self.algo_losses_ood = self._get_transformer_and_algo_outs("eval")
        
        self.train_batches, self.id_batches, self.ood_batches = self._create_batches()
            
    def _get_transformer_and_algo_outs(self, eval_mode):
        """Prepare tensors for a batch of rows."""
        algos = self.algos.copy()
        if "optimal_constant_baseline" in self.baseline_lst and "optimal_constant" not in algos:
            algos = algos + ["optimal_constant"]
        
        # Get transformer outputs
        transformer_out_col = f"{eval_mode}_outputs"
        transformer_outs = []
        
        for _, row in self.merged_df.iterrows():
            if self.loss_type != "MSE":
                out = torch.softmax(torch.as_tensor(row[transformer_out_col], device=self.device, dtype=torch.float), dim=-1)
            else:
                out = torch.as_tensor(row[transformer_out_col], device=self.device, dtype=torch.float)
            
            if self.remove_last_prediction:
                out = out[..., :-1, :]
            transformer_outs.append(out)
        
        # Prepare algorithm outputs and losses using dictionary format
        algo_outs = {}
        algo_losses = {}
        
        for algo in algos:
            algo_out_col = f"{algo}_{eval_mode}_outputs"
            
            algo_outputs = []
            algo_losses_list = []
            
            for _, row in self.merged_df.iterrows():
                if self.loss_type != "MSE":
                    out = torch.softmax(torch.as_tensor(row[algo_out_col], device=self.device, dtype=torch.float), dim=-1)
                else:
                    out = torch.as_tensor(row[algo_out_col], device=self.device, dtype=torch.float)
                
                if self.remove_last_prediction:
                    out = out[..., :-1, :]
                
                algo_outputs.append(out)
                
                # For losses, use appropriate data based on eval_mode
                if eval_mode == "train":
                    loss = torch.as_tensor(row[f"{algo}_id_all_nll"], dtype=torch.float)
                else:
                    loss = torch.as_tensor(row[f"{algo}_ood_all_nll"], dtype=torch.float)
                
                algo_losses_list.append(loss)
            
            algo_outs[algo] = algo_outputs
            algo_losses[algo] = algo_losses_list
        
        return transformer_outs, algo_outs, algo_losses
    
    def _create_batches(self):
        """Create batches for training efficiently."""
        train_batches = []
        id_batches = []
        ood_batches = []
        original_indices = []  # Store original transformer_df indices
        
        for i, (idx, row) in enumerate(self.merged_df.iterrows()):
            id_batch = self._create_batch(i, "id", row)
            ood_batch = self._create_batch(i, "ood", row)
            id_batches.append(id_batch)
            ood_batches.append(ood_batch)

            # Find corresponding index in original transformer_df
            original_idx = self.transformer_df[
                (self.transformer_df["checkpoint"] == row["checkpoint"]) &
                (self.transformer_df["num_tasks"] == row["num_tasks"]) &
                (self.transformer_df["num_dims"] == row["num_dims"]) &
                (self.transformer_df["context_length"] == row["context_length"]) &
                (self.transformer_df["mlp_expansion_factor"] == row["mlp_expansion_factor"])
            ].index[0]
            original_indices.append(original_idx)

            if row["checkpoint"] in self.train_checkpoints and row["num_tasks"] in self.train_num_tasks:
                if self.fit_with_ood:
                    train_batch = id_batch.copy()
                    train_batch["algo_outs"] = torch.cat([train_batch["algo_outs"], ood_batch["algo_outs"]], dim=0) # [B, context_length, num_dims, num_algos]
                    train_batch["algo_losses"] = torch.cat([train_batch["algo_losses"], ood_batch["algo_losses"]], dim=0) # [B, context_length, num_algos]
                    train_batch["transformer_outs"] = torch.cat([train_batch["transformer_outs"], ood_batch["transformer_outs"]], dim=0) # [B, context_length, num_dims]
                else:
                    train_batch = id_batch
                
                train_batches.append(train_batch)
        
        self.original_indices = original_indices  # Store for later use
        return train_batches, id_batches, ood_batches
    
    def _create_batch(self, i, eval_mode, row):
        """Create a batch dictionary for evaluation efficiently."""
        algo_outs = torch.stack([self.algo_outs_id[algo][i] for algo in self.algos], dim=-1) if eval_mode == "id" \
            else torch.stack([self.algo_outs_ood[algo][i] for algo in self.algos], dim=-1) # [B, context_length, num_dims, num_algos]
        if len(algo_outs.shape) == 3:
            algo_outs = algo_outs.unsqueeze(-2) # [B, context_length, 1, num_algos]
        
        algo_losses = torch.stack([self.algo_losses_id[algo][i] for algo in self.algos], dim=-1) if eval_mode == "id" \
            else torch.stack([self.algo_losses_ood[algo][i] for algo in self.algos], dim=-1) 
        
        transformer_out = self.transformer_outs_id[i] if eval_mode == "id" else self.transformer_outs_ood[i]
        if len(transformer_out.shape) == 2:
            transformer_out = transformer_out.unsqueeze(-1) # [B, context_length, 1]
            
        batch_dict = {
            'algo_outs': algo_outs,
            'algo_losses': algo_losses,
            'transformer_outs': transformer_out,  # [B, context_length, num_dims]
            'exp_params': {
                "checkpoint": row["checkpoint"],
                "num_tasks": row["num_tasks"],
                "num_dims": row["num_dims"],
                "context_length": row["context_length"],
            }
        }
        return batch_dict
        
    def _create_model(self):
        """Create the HierarchicalBayesianModel model."""
        if not self.ablation_model:
            self.model = HierarchicalBayesianModel(
                self.params_init, self.complexity_lookup, self.avg_losses_lookup, 
                self.metric_name, self.context_dependent_weights, self.batch_size)
        else:
            self.model = HierarchicalBayesianModelAblated(
                self.params_init, self.complexity_lookup, self.avg_losses_lookup, self.metric_name, self.context_dependent_weights, 
                linear_likelihood=self.ablation_model_args["linear_likelihood"], 
                fixed_complexity=self.ablation_model_args["fixed_complexity"], 
                no_random_loss_term=self.ablation_model_args["no_random_loss_term"],
                linear_param_on_complexity=self.ablation_model_args["linear_param_on_complexity"],
                batch_size=self.batch_size
            )

        if self.fit_params:
            device = get_device()
            self.model = self.model.to(device)
            
    
    def _train_model(self):
        """Train the model with prepared batches."""
        if not self.fit_params:
            self.history = []
            return
        
        # Set fixed parameters
        for param in self.fixed_params:
            self.model.params[param].requires_grad = False
        
        # Check if there are any parameters that still require gradients
        if not any(param.requires_grad for param in self.model.parameters()):
            print("All parameters are frozen.")
            self.history = []
            return
        
        
        print(f"Training HierarchicalBayesianModel model with {len(self.train_batches)} batches...")
        self.history = ScipyOptimizer.train_model(
            self.model, self.train_batches, 
            method=self.method, 
            ftol=self.ftol, 
            gtol=self.gtol, 
            return_history=self.return_history
        )
    
    def _evaluate_and_save(self):
        """Evaluate model and save results for all data points using vectorized operations."""
        self.model.eval()

        # Initialize baseline computer with prepared data
        if self.baseline_lst:
            self.baseline_computer = self.BaselineComputer(self)

        with torch.no_grad():
            for i, (id_batch, ood_batch) in tqdm_func()(enumerate(zip(self.id_batches, self.ood_batches)), total=len(self.id_batches)):

                # get row index and data using stored original indices
                original_idx = self.original_indices[i]
                row = self.transformer_df.loc[original_idx]

                id_batch = move_batch_to_device(id_batch, get_device())
                ood_batch = move_batch_to_device(ood_batch, get_device())

                # get model outputs
                outputs_id = self.model(**id_batch)
                outputs_ood = self.model(**ood_batch)
                
                device = outputs_id["weights"].device
                
                # compute evaluation metrics
                results = self._compute_evaluation_metrics(outputs_id, outputs_ood, i, device)
                
                # handle baselines
                baseline_results = self.baseline_computer.compute_all_baselines(i) if self.baseline_lst else {}

                # extract model outputs
                weights_id = outputs_id["weights"].cpu()
                log_prior_odds_id = outputs_id["log_prior_odds"].cpu()
                log_bayes_factor_id = outputs_id["log_bayes_factor"].cpu()
                log_posterior_odds_id = outputs_id["log_posterior_odds"].cpu()
                weights_ood = outputs_ood["weights"].cpu()
                log_prior_odds_ood = outputs_ood["log_prior_odds"].cpu()
                log_bayes_factor_ood = outputs_ood["log_bayes_factor"].cpu()
                log_posterior_odds_ood = outputs_ood["log_posterior_odds"].cpu()

                # create results dictionary
                bms_results = {
                    "weights": weights_id.mean(dim=(0, 1)).tolist(),
                    "ood_weights": weights_ood.mean(dim=(0, 1)).tolist(),
                    "log_prior_odds_id": log_prior_odds_id.mean().item(),
                    "log_bayes_factor_id": log_bayes_factor_id.mean().item(),
                    "log_posterior_odds_id": log_posterior_odds_id.mean().item(),
                    "log_prior_odds_ood": log_prior_odds_ood.mean().item(),
                    "log_bayes_factor_ood": log_bayes_factor_ood.mean().item(),
                    "log_posterior_odds_ood": log_posterior_odds_ood.mean().item(),
                    "seen": True if row["checkpoint"] in self.train_checkpoints and row["num_tasks"] in self.train_num_tasks else False,
                    "results": results,
                    "baseline_results": baseline_results if self.baseline_lst else None,
                    "params": {name: param.item() for name, param in outputs_id["params"].items()},
                    "metric_name": self.metric_name,
                }
                
                # save to DataFrame using original index
                if self.add_to_df:
                    self.transformer_df.at[original_idx, self.out_column] = bms_results
                
                # save to file
                self._save_results_to_file(row, bms_results)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _compute_evaluation_metrics(self, outputs_id, outputs_ood, i, device):
        """Compute evaluation metrics comparing model interpolation to transformer outputs."""
        results = {}
        
        if self.metric_name == "MSE":
            results["r_squared"] = {
                "id": r2_score(self.transformer_outs_id[i].cpu().numpy().ravel(), outputs_id["interpolated_out"].cpu().numpy().ravel()),
                "ood": r2_score(self.transformer_outs_ood[i].cpu().numpy().ravel(), outputs_ood["interpolated_out"].cpu().numpy().ravel())
            }
            results["mse"] = {
                "id": np.mean((self.transformer_outs_id[i].cpu().numpy().ravel() - outputs_id["interpolated_out"].cpu().numpy().ravel())**2).item(),
                "ood": np.mean((self.transformer_outs_ood[i].cpu().numpy().ravel() - outputs_ood["interpolated_out"].cpu().numpy().ravel())**2).item()
            }
        elif self.metric_name == "KL":
            results["total_variation_distance"] = {
                "id": total_variation_distance(outputs_id["interpolated_out"].to(device), self.transformer_outs_id[i].to(device)).item(),
                "ood": total_variation_distance(outputs_ood["interpolated_out"].to(device), self.transformer_outs_ood[i].to(device)).item()
            }
            
            if self.setting == "classification":
                results["agreement"] = {
                    "id": (outputs_id["interpolated_out"].to(device).argmax(dim=-1) == self.transformer_outs_id[i].to(device).argmax(dim=-1)).float().mean().item(),
                    "ood": (outputs_ood["interpolated_out"].to(device).argmax(dim=-1) == self.transformer_outs_ood[i].to(device).argmax(dim=-1)).float().mean().item()
                }
            elif self.setting == "categorical-sequence":
                results["spearman_correlation"] = {
                    "id": batched_spearman_correlation(outputs_id["interpolated_out"].to(device), self.transformer_outs_id[i].to(device)).item(),
                    "ood": batched_spearman_correlation(outputs_ood["interpolated_out"].to(device), self.transformer_outs_ood[i].to(device)).item()
                }
        
        return results
    
    def _save_results_to_file(self, row, bms_results):
        """Save results to file."""
        output_dir = row["config"].output_dir
        if not self.ablation_model:
            context_dependent_suffix = "_context_dependent" if self.context_dependent_weights else ""
            bms_results_file = os.path.join(output_dir, f"checkpoint-{row['checkpoint']}", f"bms_results{context_dependent_suffix}.json")
        else:
            bms_results_file = os.path.join(output_dir, f"checkpoint-{row['checkpoint']}", f"{self.out_column}.json")
        
        with open(bms_results_file, "w") as f:
            json.dump(bms_results, f)
    
    def _write_average_evaluation_results(self):
        """Write correlation results to file."""
        if self.ablation_model:
            return
        
        # Compute correlations
        query_str = f"context_length == {self.context_length} & num_dims == {self.num_dims} & mlp_expansion_factor == {self.mlp_expansion_factor}"
        filtered_results = self.transformer_df.query(query_str, engine="python")
        
        relative_distance_predicted_id = filtered_results[self.out_column].apply(lambda x: x["weights"][0]).values.astype(np.float64)
        relative_distance_predicted_ood = filtered_results[self.out_column].apply(lambda x: x["ood_weights"][0]).values.astype(np.float64)
        relative_distance_observed_id = filtered_results["relative_distance_train"].values.astype(np.float64)
        relative_distance_observed_ood = filtered_results["relative_distance_eval"].values.astype(np.float64)
                
        correlation_bms_relative_distance_id = np.corrcoef(relative_distance_predicted_id, relative_distance_observed_id)[0, 1].item()
        correlation_bms_relative_distance_ood = np.corrcoef(relative_distance_predicted_ood, relative_distance_observed_ood)[0, 1].item()
        
        # Record average values for results across different conditions
        average_results = {}
        if self.metric_name == "MSE":
            average_results["r_squared_id"] = filtered_results[self.out_column].apply(lambda x: x["results"]["r_squared"]["id"]).mean().item()
            average_results["r_squared_ood"] = filtered_results[self.out_column].apply(lambda x: x["results"]["r_squared"]["ood"]).mean().item()
        elif self.metric_name == "KL":
            average_results["total_variation_distance_id"] = filtered_results[self.out_column].apply(lambda x: x["results"]["total_variation_distance"]["id"]).mean().item()
            average_results["total_variation_distance_ood"] = filtered_results[self.out_column].apply(lambda x: x["results"]["total_variation_distance"]["ood"]).mean().item()
            
            if self.setting == "categorical-sequence":
                average_results["spearman_correlation_id"] = filtered_results[self.out_column].apply(lambda x: x["results"]["spearman_correlation"]["id"]).mean().item()
                average_results["spearman_correlation_ood"] = filtered_results[self.out_column].apply(lambda x: x["results"]["spearman_correlation"]["ood"]).mean().item()
            elif self.setting == "classification":
                average_results["agreement_id"] = filtered_results[self.out_column].apply(lambda x: x["results"]["agreement"]["id"]).mean().item()
                average_results["agreement_ood"] = filtered_results[self.out_column].apply(lambda x: x["results"]["agreement"]["ood"]).mean().item()
        
        # Write to correlation file
        correlation_file = os.path.join(here("results"), f"{self.transformer_df.iloc[0]['config'].setting}_correlations.jsonl")
        new_entry = {
            "context_length": self.context_length, 
            "num_dims": self.num_dims, 
            "mlp_expansion_factor": self.mlp_expansion_factor, 
            "correlation_bms_relative_distance_id": correlation_bms_relative_distance_id, 
            "correlation_bms_relative_distance_ood": correlation_bms_relative_distance_ood,
            "average_results": average_results
        }
        
        # Handle file writing with duplicate checking
        existing_entries = []
        found_existing = False
        
        if os.path.exists(correlation_file):
            with open(correlation_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if (entry.get("context_length") == self.context_length and 
                        entry.get("num_dims") == self.num_dims and 
                        entry.get("mlp_expansion_factor") == self.mlp_expansion_factor):
                        found_existing = True
                        continue
                    existing_entries.append(line)
            
            if found_existing:
                with open(correlation_file, "w") as f:
                    for line in existing_entries:
                        f.write(line)
                    f.write(json.dumps(new_entry) + "\n")
            else:
                with open(correlation_file, "a") as f:
                    f.write(json.dumps(new_entry) + "\n")
        else:
            os.makedirs(os.path.dirname(correlation_file), exist_ok=True)
            with open(correlation_file, "w") as f:
                f.write(json.dumps(new_entry) + "\n")

    class BaselineComputer:
        """
        A nested class to compute various baseline results for model evaluation.
        
        This class encapsulates all baseline computation methods and pre-loads
        random transformer data if needed to avoid repeated loading during evaluation.
        """
        
        def __init__(self, parent):
            """
            Initialize the baseline computer.
            
            Args:
                parent: The parent HierarchicalBayesianModelFitter instance
            """
            self.parent = parent
            
            # Pre-load random transformer data if needed
            self.random_transformer_outs_id = None
            self.random_transformer_outs_ood = None
            if "random_transformer_baseline" in self.parent.baseline_lst:
                self._preload_random_transformer_data()
        
        def _preload_random_transformer_data(self):
            """Pre-load random transformer outputs to avoid repeated loading during evaluation."""
            self.random_transformer_outs_id = []
            self.random_transformer_outs_ood = []
            
            for _, row in self.parent.merged_df.iterrows():
                # Load random transformer outputs for ID data
                if self.parent.loss_type != "MSE":
                    out_id = torch.softmax(torch.as_tensor(row["train_outputs-random"], device=self.parent.device, dtype=torch.float), dim=-1)
                    out_ood = torch.softmax(torch.as_tensor(row["eval_outputs-random"], device=self.parent.device, dtype=torch.float), dim=-1)
                else:
                    out_id = torch.as_tensor(row["train_outputs-random"], device=self.parent.device, dtype=torch.float)
                    out_ood = torch.as_tensor(row["eval_outputs-random"], device=self.parent.device, dtype=torch.float)
                
                if self.parent.remove_last_prediction:
                    out_id = out_id[..., :-1, :]
                    out_ood = out_ood[..., :-1, :]
                
                self.random_transformer_outs_id.append(out_id)
                self.random_transformer_outs_ood.append(out_ood)
        
        def compute_all_baselines(self, i):
            """
            Compute all requested baseline results for a given index.
            
            Args:
                i: Index of the data point to compute baselines for
                
            Returns:
                dict: Dictionary containing all baseline results
            """
            baseline_results = {}
            
            if "random_weights_baseline" in self.parent.baseline_lst:
                baseline_results["random_weights_baseline"] = self.compute_random_weights_baseline(i)
            
            if "random_transformer_baseline" in self.parent.baseline_lst:
                baseline_results["random_transformer_baseline"] = self.compute_random_transformer_baseline(i)
            
            if "optimal_constant_baseline" in self.parent.baseline_lst:
                baseline_results["optimal_constant_baseline"] = self.compute_optimal_constant_baseline(i)
            
            return baseline_results
        
        def compute_random_transformer_baseline(self, i):
            """Compute random transformer baseline efficiently."""
            # Use pre-loaded random transformer outputs
            out_id = self.random_transformer_outs_id[i]
            out_ood = self.random_transformer_outs_ood[i]

            baseline_results = {}
            if self.parent.metric_name == "MSE":
                baseline_results["r_squared"] = {
                    "id": r2_score(out_id.cpu().numpy().ravel(), self.parent.transformer_outs_id[i].cpu().numpy().ravel()),
                    "ood": r2_score(out_ood.cpu().numpy().ravel(), self.parent.transformer_outs_ood[i].cpu().numpy().ravel())
                }
            elif self.parent.metric_name == "KL":
                baseline_results["total_variation_distance"] = {
                    "id": total_variation_distance(out_id.to(self.parent.device), self.parent.transformer_outs_id[i].to(self.parent.device)).item(),
                    "ood": total_variation_distance(out_ood.to(self.parent.device), self.parent.transformer_outs_ood[i].to(self.parent.device)).item()
                }
            
            return baseline_results
        
        def compute_random_weights_baseline(self, i):
            """Compute random weights baseline efficiently."""
            # draw a distribution that adds up to 1
            random_dist = torch.rand((1, 1, len(self.parent.algos)), device=self.parent.device)
            random_dist = random_dist / random_dist.sum() 
            
            # Stack algorithm outputs efficiently
            algo_outs_id_stacked = torch.stack([self.parent.algo_outs_id[algo][i] for algo in self.parent.algos], dim=-1)  # [context_length, num_dims, num_algos]
            algo_outs_ood_stacked = torch.stack([self.parent.algo_outs_ood[algo][i] for algo in self.parent.algos], dim=-1)  # [context_length, num_dims, num_algos]
            
            # Weighted average across algorithms (last dimension)
            interpolation_random_baseline_id = (algo_outs_id_stacked * random_dist).sum(dim=-1)
            interpolation_random_baseline_ood = (algo_outs_ood_stacked * random_dist).sum(dim=-1)
            
            baseline_results = {}
            if self.parent.metric_name == "MSE":
                baseline_results["r_squared"] = {
                    "id": r2_score(self.parent.transformer_outs_id[i].cpu().numpy().ravel(), interpolation_random_baseline_id.cpu().numpy().ravel()),
                    "ood": r2_score(self.parent.transformer_outs_ood[i].cpu().numpy().ravel(), interpolation_random_baseline_ood.cpu().numpy().ravel())
                }
                baseline_results["mse"] = {
                    "id": np.mean((self.parent.transformer_outs_id[i].cpu().numpy().ravel() - interpolation_random_baseline_id.cpu().numpy().ravel())**2).item(),
                    "ood": np.mean((self.parent.transformer_outs_ood[i].cpu().numpy().ravel() - interpolation_random_baseline_ood.cpu().numpy().ravel())**2).item()
                }
            elif self.parent.metric_name == "KL":  
                baseline_results["total_variation_distance"] = {
                    "id": total_variation_distance(interpolation_random_baseline_id.to(self.parent.device), self.parent.transformer_outs_id[i].to(self.parent.device)).item(),
                    "ood": total_variation_distance(interpolation_random_baseline_ood.to(self.parent.device), self.parent.transformer_outs_ood[i].to(self.parent.device)).item()
                }
            
            return baseline_results
        
        def compute_optimal_constant_baseline(self, i):
            """Compute optimal constant baseline efficiently."""
            baseline_results = {}
            
            # Only compute if optimal_constant is in algorithms
            if "optimal_constant" in self.parent.algo_outs_id.keys():
                # Extract optimal_constant outputs from dictionary
                optimal_constant_id = self.parent.algo_outs_id["optimal_constant"][i]
                optimal_constant_ood = self.parent.algo_outs_ood["optimal_constant"][i]
                
                if self.parent.metric_name == "MSE":
                    baseline_results["r_squared"] = {
                        "id": r2_score(self.parent.transformer_outs_id[i].cpu().numpy().ravel(), optimal_constant_id.cpu().numpy().ravel()),
                        "ood": r2_score(self.parent.transformer_outs_ood[i].cpu().numpy().ravel(), optimal_constant_ood.cpu().numpy().ravel())
                    }
                elif self.parent.metric_name == "KL":
                    baseline_results["total_variation_distance"] = {
                        "id": total_variation_distance(optimal_constant_id.to(self.parent.device), self.parent.transformer_outs_id[i].to(self.parent.device)).item(),
                        "ood": total_variation_distance(optimal_constant_ood.to(self.parent.device), self.parent.transformer_outs_ood[i].to(self.parent.device)).item()
                    }
            
            return baseline_results

##########################
# Visualization helper functions
##########################

def format_axis_labels_as_powers(
    ax, 
    axis='y',
    base=2,
    base_x_pos=-0.105,
    base_y_offset=-0.1,
    power_x_offset=-0.001,
    power_y_offset=0.4,
    base_fontname='Avenir',
    base_fontweight='light',
    base_fontsize=22,
    power_fontsize=16,
    only_show_even_powers=True
):
    """
    Format axis labels as powers (e.g., 2^4 instead of 16).
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to modify
    axis : str, optional
        Which axis to modify ('x' or 'y'), default 'y'
    base : int, optional
        The base to use (e.g., 2 or 10), default 2
    base_x_pos : float, optional
        X position of the base text relative to the axis, default -0.105
    base_y_offset : float, optional
        Y offset for the base text, default -0.1
    power_x_offset : float, optional
        X offset for the power text relative to the base, default -0.001
    power_y_offset : float, optional
        Y offset for the power text, default 0.4
    base_fontname : str, optional
        Font name for the base, default 'Avenir'
    base_fontweight : str, optional
        Font weight for the base, default 'light'
    base_fontsize : int, optional
        Font size for the base, default 22
    power_fontsize : int, optional
        Font size for the power, default 16
    only_show_even_powers : bool, optional
        If True, only show labels for even powers, default True
    
    Returns:
    --------
    None
    """
    # Determine which axis we're working with
    if axis.lower() == 'y':
        get_ticks = ax.get_yticks
        get_ticklabels = ax.get_yticklabels
        set_ticklabels = ax.set_yticklabels
        transform = ax.get_yaxis_transform()
        ha_base = 'right'
        ha_power = 'left'
        va = 'center'
        va_power = 'top'
    elif axis.lower() == 'x':
        get_ticks = ax.get_xticks
        get_ticklabels = ax.get_xticklabels
        set_ticklabels = ax.set_xticklabels
        transform = ax.get_xaxis_transform()
        ha_base = 'center'
        ha_power = 'left'
        va = 'top'
        va_power = 'bottom'
    else:
        raise ValueError("axis must be 'x' or 'y'")
    
    # Get the positions of the ticks in data coordinates
    positions = get_ticks()
    
    # Calculate powers based on tick labels
    try:
        # Try to get powers from tick labels (if they're numbers)
        powers = [int(np.log(float(label.get_text())) / np.log(base)) 
                 for label in get_ticklabels() if label.get_text().strip() and label.get_text() != " "]
    except (ValueError, AttributeError):
        # If that fails, calculate powers directly from positions
        powers = [int(np.log(pos) / np.log(base)) if pos > 0 else 0 for pos in positions]
    
    # Clear existing labels
    set_ticklabels([" " for _ in positions])  # hide default labels
    
    # Add custom labels with styled superscripts
    for pos, power in zip(positions, powers):
        # Skip odd powers if only_show_even_powers is True
        if only_show_even_powers and power % 2 != 0:
            continue
            
        # For y-axis, we use the position directly
        # For x-axis, we need to adjust for the transform
        if axis.lower() == 'y':
            y_pos = pos + base_y_offset
            power_y = pos + power_y_offset
        else:
            y_pos = base_y_offset
            power_y = power_y_offset
            
        # Add the base
        ax.text(
            x=base_x_pos if axis.lower() == 'y' else pos + base_y_offset,
            y=y_pos if axis.lower() == 'y' else base_x_pos,
            s=str(base),
            fontname=base_fontname,
            fontweight=base_fontweight,
            fontsize=base_fontsize,
            va=va,
            ha=ha_base,
            transform=transform
        )
        
        # Add the power
        ax.text(
            x=(base_x_pos + power_x_offset) if axis.lower() == 'y' else pos + power_y_offset,
            y=power_y if axis.lower() == 'y' else (base_x_pos + power_x_offset),
            s=str(power),
            fontname=base_fontname,
            fontweight=base_fontweight,
            fontsize=power_fontsize,
            va=va_power,
            ha=ha_power,
            transform=transform
        )


def format_axis_labels_in_thousands(ax, axis='x'):
    """
    Format tick labels to rounded values with K suffix (for thousands).
    For values ≥ 5000: Round to nearest 5K
    For values between 1000-5000: Round to nearest 1K
    For values < 1000: Round to nearest 0.25K
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to format
    axis : str, optional
        Which axis to format ('x', 'y', or 'both')
    """
    def format_tick_value(val):
        try:
            val = float(val)
        except (ValueError, TypeError):
            return ""  # Return empty string if it can't be converted
            
        abs_val = abs(val)
        sign = -1 if val < 0 else 1
        
        if abs_val >= 5000:
            # Round to nearest 5K
            rounded = round(abs_val / 5000) * 5000 * sign
            return f"{rounded/1000:.0f}K"
        elif abs_val >= 1000:
            # Round to nearest 1K
            rounded = round(abs_val / 1000) * 1000 * sign
            return f"{rounded/1000:.0f}K"
        elif abs_val >= 50:
            # Round to nearest 0.1K
            rounded = round(abs_val / 100) * 100 * sign
            return f"{rounded/1000:.1f}K"
        else:
            return "0"
    
    if axis in ['x', 'both']:
        # Get the current tick labels
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        # Format the values
        xticklabels = [format_tick_value(val) for val in x_labels]
        ax.set_xticklabels(xticklabels)
    
    if axis in ['y', 'both']:
        # Get the current tick labels
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        # Format the values
        yticklabels = [format_tick_value(val) for val in y_labels]
        ax.set_yticklabels(yticklabels)


def plot_flexible_scatterplot(
    df,
    variable_config,
    fixed_values,
    value_column,
    needed_cols_for_callable=None,
    x_scale='linear',
    x_scale_base=10,  
    y_scale='linear',
    y_scale_base=10,  
    y_label=None,
    x_label=None,
    title=None,
    color='blue',
    figsize=(10, 6),
    marker='o',
    linestyle='-',
    linewidth=2,
    y_lim=None,
    cmap=None,
    legend_labels=None,
    legend_title=None,
    legend_loc='best',
    show_title=True,
    frameon=True,
    markersize=None,
    title_fontsize=14,
    axis_label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    legend_title_fontsize=12,
    ax=None
):
    """
    Plot a flexible scatterplot for given fixed variables and an optional grid variable.
    
    Args:
        df: DataFrame containing the data
        variable_config: Dictionary with keys 'fixed', 'iterate_over', and 'grid_by'
        fixed_values: Dictionary of fixed variable values (e.g., {'context_length': 128, 'mlp_expansion_factor': 4})
        value_column: Column to plot on y-axis, can be a string, callable function, or list of strings/callables
        needed_cols_for_callable: List of column names needed if value_column is a callable, or list of lists if
                                 value_column is a list of callables
        x_scale: Scale for x-axis ('log', 'linear', etc.)
        x_scale_base: Base for logarithmic scale (default: 10, only used when x_scale='log')
        x_label: Label for x-axis (defaults to iterate_over if string)
        y_scale: Scale for y-axis ('log', 'linear', etc.)
        y_scale_base: Base for logarithmic scale (default: 10, only used when y_scale='log')
        y_label: Label for y-axis (defaults to value_column if string), can be a list matching value_column
        title: Title for the plot (set to None for no title)
        color: Color for the plot, can be a single color or list of colors matching value_column
        figsize: Figure size as (width, height)
        marker: Marker style for points, can be a single marker or list matching value_column
        linestyle: Line style, can be a single style or list matching value_column
        linewidth: Line width, can be a single width or list matching value_column
        y_lim: Tuple of (min, max) for y-axis limits
        cmap: Name of matplotlib colormap to use for grid values (e.g., 'viridis', 'plasma', 'tab10')
        legend_labels: Custom labels for the legend (list of strings)
        legend_title: Custom title for the legend
        legend_loc: Location for the legend ('best', 'upper right', etc.)
        show_title: Whether to show the title (if provided)
        frameon: Whether to show a frame around the legend
        markersize: Size of markers (if None, uses matplotlib default)
        title_fontsize: Font size for the plot title (default: 14)
        axis_label_fontsize: Font size for axis labels (default: 12)
        tick_fontsize: Font size for tick labels (default: 10)
        legend_fontsize: Font size for legend text (default: 10)
        legend_title_fontsize: Font size for legend title (default: 12)
        ax: Matplotlib axis object (default: None)
    
    Returns:
        matplotlib figure object
    """
    # Validate variable_config
    valid_variables = {"context_length", "num_tasks", "num_dims", "mlp_expansion_factor", "checkpoint"}
    used_variables = set(variable_config["fixed"]) | {variable_config["iterate_over"]}
    if variable_config["grid_by"]:
        used_variables.add(variable_config["grid_by"])
    for var in used_variables:
        if var not in valid_variables:
            raise ValueError(f"Use only the following variables as fixed variables: {valid_variables}!")

    # Filter the dataframe based on fixed values
    filtered_df = df.copy()
    for var in variable_config["fixed"]:
        if var in fixed_values:
            filtered_df = filtered_df[filtered_df[var] == fixed_values[var]]
    
    if filtered_df.empty:
        print(f"No data found for fixed values {fixed_values}")
        return None
    
    # Determine x-axis variable
    x_variable = variable_config["iterate_over"]
    
    # Convert single value_column to list for uniform processing
    value_columns = value_column if isinstance(value_column, list) else [value_column]
    num_columns = len(value_columns)
    
    # Convert other parameters to lists if they're not already
    colors = color if isinstance(color, list) else [color] * num_columns
    markers = marker if isinstance(marker, list) else [marker] * num_columns
    linestyles = linestyle if isinstance(linestyle, list) else [linestyle] * num_columns
    linewidths = linewidth if isinstance(linewidth, list) else [linewidth] * num_columns
    y_labels = y_label if isinstance(y_label, list) else [y_label] * num_columns
    
    # Ensure all lists have the same length as value_columns
    if len(colors) < num_columns:
        colors = colors * (num_columns // len(colors) + 1)
    if len(markers) < num_columns:
        markers = markers * (num_columns // len(markers) + 1)
    if len(linestyles) < num_columns:
        linestyles = linestyles * (num_columns // len(linestyles) + 1)
    if len(linewidths) < num_columns:
        linewidths = linewidths * (num_columns // len(linewidths) + 1)
    if len(y_labels) < num_columns:
        y_labels = y_labels * (num_columns // len(y_labels) + 1)
    
    # Process needed columns for callables
    if needed_cols_for_callable is not None:
        # Convert to list of lists if it's a flat list
        if not any(isinstance(item, list) for item in needed_cols_for_callable):
            needed_cols_for_callable = [needed_cols_for_callable] * num_columns
        
        # Ensure the list has the right length
        if len(needed_cols_for_callable) < num_columns:
            needed_cols_for_callable = needed_cols_for_callable * (num_columns // len(needed_cols_for_callable) + 1)
        
        # Process each set of needed columns
        for col_set in needed_cols_for_callable:
            if col_set is None:
                continue
            for col in col_set:
                if col not in filtered_df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                # Filter out NaNs in needed columns
                filtered_df = filtered_df[filtered_df[col].notna()]
    
    # Calculate y values for each column
    y_columns = []
    for i, val_col in enumerate(value_columns):
        if callable(val_col):
            col_name = f'plot_value_{i}'
            filtered_df[col_name] = filtered_df.apply(val_col, axis=1)
            y_columns.append(col_name)
        else:
            # Ensure the column exists
            if val_col not in filtered_df.columns:
                raise ValueError(f"Column '{val_col}' not found in DataFrame")
            y_columns.append(val_col)
    
    # Convert to numeric and handle non-numeric values for each y column
    for y_col in y_columns:
        filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')
    
    # Drop rows with NaN in any y column
    filtered_df = filtered_df.dropna(subset=y_columns)
    
    # Create figure
    # define fig, ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Store plot lines for legend
    plot_lines = []
    
    # Plot for each grid value or a single line if grid_variable is None
    if variable_config["grid_by"]:
        grid_values = sorted(filtered_df[variable_config["grid_by"]].unique())
        
        # Set up color palette
        if cmap is None:
            cmap = 'viridis'  # Default colormap
        
        # Create colormap for grid values
        cmap = plt.cm.get_cmap(cmap, len(grid_values))
        grid_colors = [cmap(i) for i in range(len(grid_values))]
        
        for y_idx, y_col in enumerate(y_columns):
            for i, grid_val in enumerate(grid_values):
                grid_df = filtered_df[filtered_df[variable_config["grid_by"]] == grid_val]
                grid_df = grid_df.sort_values(x_variable)
                
                # Use color from palette for this grid value
                plot_color = grid_colors[i]
                
                # Format grid value for display
                if variable_config["grid_by"] == "num_tasks":
                    display_val = grid_val
                elif variable_config["grid_by"] == "checkpoint":
                    display_val = f"{grid_val:,}"
                else:
                    display_val = grid_val
                
                # Simplified label that only shows the grid value
                label = f'{display_val}'
                
                line, = ax.plot(
                    grid_df[x_variable], 
                    grid_df[y_col], 
                    marker=markers[y_idx], 
                    linestyle=linestyles[y_idx], 
                    color=plot_color,
                    linewidth=linewidths[y_idx],
                    label=label,
                    markersize=markersize
                )
                plot_lines.append(line)
    else:
        filtered_df = filtered_df.sort_values(x_variable)
        
        # Use custom legend labels if provided
        if legend_labels is not None and len(legend_labels) >= num_columns:
            display_labels = legend_labels[:num_columns]
        else:
            display_labels = []
            for y_idx, y_col in enumerate(y_columns):
                if y_labels[y_idx]:
                    display_labels.append(y_labels[y_idx])
                elif isinstance(value_columns[y_idx], str):
                    display_labels.append(value_columns[y_idx])
                else:
                    display_labels.append(f'Series {y_idx+1}')
        
        for y_idx, y_col in enumerate(y_columns):
            line, = ax.plot(
                filtered_df[x_variable], 
                filtered_df[y_col], 
                marker=markers[y_idx], 
                linestyle=linestyles[y_idx], 
                color=colors[y_idx],
                linewidth=linewidths[y_idx],
                label=display_labels[y_idx],
                markersize=markersize
            )
            plot_lines.append(line)
    
    # Configure plot
    if x_scale == 'log':
        ax.set_xscale('log', base=x_scale_base)
    else:
        ax.set_xscale(x_scale)

    if y_scale == 'log':
        ax.set_yscale('log', base=y_scale_base)
    else:
        ax.set_yscale(y_scale)
    
    # Format x-axis label
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    else:
        ax.set_xlabel(x_variable.replace('_', ' ').title(), fontsize=axis_label_fontsize)
    
    # Set y-axis label - use first non-None label or generic
    if any(label is not None for label in y_labels):
        first_valid_label = next((label for label in y_labels if label is not None), 'Value')
        ax.set_ylabel(first_valid_label, fontsize=axis_label_fontsize)
    else:
        # Use the grid_by variable in the y-label if it's a metric
        if variable_config["grid_by"]:
            grid_var_name = variable_config["grid_by"].replace('_', ' ').title()
            if y_columns[0].endswith('metric'):
                ax.set_ylabel(f'Evaluation Metric', fontsize=axis_label_fontsize)
            else:
                ax.set_ylabel('Value', fontsize=axis_label_fontsize)
        else:
            ax.set_ylabel('Value', fontsize=axis_label_fontsize)
    
    if y_lim:
        ax.set_ylim(y_lim)
    
    # Set title if requested
    if title and show_title:
        ax.set_title(title, fontsize=title_fontsize)
    elif show_title and not title:
        # Create a more descriptive title
        fixed_str = ', '.join([f"{k.replace('_', ' ').title()}={v}" for k, v in fixed_values.items()])
        if variable_config["grid_by"]:
            grid_var_name = variable_config["grid_by"].replace('_', ' ').title()
            title = f'Training Trajectories by {grid_var_name}'
        else:
            title = f'Values for {fixed_str}'
        ax.set_title(title, fontsize=title_fontsize)
    
    # Set tick font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Add legend with custom settings
    if len(plot_lines) > 0:
        if variable_config["grid_by"]:
            # Use grid variable as legend title if not overridden
            if legend_title is None:
                legend_title = variable_config["grid_by"].replace('_', ' ').title()
            legend = ax.legend(title=legend_title, loc=legend_loc, frameon=frameon, fontsize=legend_fontsize)
            legend.get_title().set_fontsize(legend_title_fontsize)
        else:
            # Use custom legend title if provided
            if legend_title:
                legend = ax.legend(title=legend_title, loc=legend_loc, frameon=frameon, fontsize=legend_fontsize)
                legend.get_title().set_fontsize(legend_title_fontsize)
            else:
                ax.legend(loc=legend_loc, frameon=frameon, fontsize=legend_fontsize)        
    
    return fig, ax

def plot_flexible_heatmap(
    df,
    variable_config,
    fixed_values,
    value_columns,
    needed_cols_for_callable=None,
    ax=None,
    titles=None,
    vmins=None,
    vmaxs=None,
    cmaps=None,
    annot=True,
    fmt=".2f",
    annot_kws={'size': 6},
    reverse_x_axis=False,
    reverse_y_axis=False,
    figsize=None,
    title_fontsize=14,
    axis_label_fontsize=12,
    tick_fontsize=10,
    x_axis_label=None,
    y_axis_label=None,
    colorbar_labels=None,
    colorbar_location='right',
    grid_by_column=False  # New parameter to control layout orientation
):
    """
    Plot a grid of heatmaps for multiple specified columns or callable functions.
    
    Args:
        df: DataFrame containing the data
        variable_config: Dictionary with keys 'iterate_over' and optionally 'grid_by'
        fixed_values: Dictionary of fixed variable values
        value_columns: List of column names or callable functions to plot
        needed_cols_for_callable: List of lists, where each inner list contains column names 
                                 needed for the corresponding callable in value_columns
        ax: Optional pre-existing axes to plot on
        titles: List of titles for each heatmap column
        vmins: List of minimum values for color scaling
        vmaxs: List of maximum values for color scaling
        cmaps: List of colormaps to use
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
        annot_kws: Dictionary of keyword arguments for annotations
        reverse_x_axis: Whether to reverse the x-axis
        reverse_y_axis: Whether to reverse the y-axis
        figsize: Figure size as (width, height) per subplot
        title_fontsize: Font size for subplot titles
        axis_label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
        x_axis_label: Custom label for x-axis (overrides default)
        y_axis_label: Custom label for y-axis (overrides default)
        colorbar_labels: List of labels for colorbars
        colorbar_location: Location of the colorbar ('right' or 'bottom')
        grid_by_column: If True, grid values are shown side by side in columns and value_columns in rows.
                        If False, grid values are shown in rows and value_columns side by side.
    
    Returns:
        fig, axes: Figure and axes objects
    """
    def standardize_param_list(param, default_value, num_columns):
        if param is None:
            return [default_value] * num_columns
        if not isinstance(param, list):
            return [param] * num_columns
        return param[:num_columns] + [default_value] * (num_columns - len(param))
    
    # Helper function to prepare data for a specific column
    def prepare_data(df_filtered, value_column, needed_cols):
        current_df = df_filtered.copy()
        
        # Filter out NaN values in needed columns
        if needed_cols is not None:
            for col in needed_cols:
                if col not in current_df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                current_df = current_df[current_df[col].notna()]
        
        # Apply callable or use column directly
        if callable(value_column):
            plot_values = []
            for _, row in current_df.iterrows():
                try:
                    plot_values.append(value_column(row))
                except (TypeError, KeyError, AttributeError, IndexError):
                    plot_values.append(float('nan'))
            
            current_df['plot_value'] = plot_values
            plot_col = 'plot_value'
        else:
            plot_col = value_column
        
        # Convert to numeric and handle NaNs
        if plot_col in current_df.columns:
            current_df[plot_col] = pd.to_numeric(current_df[plot_col], errors='coerce')
            current_df = current_df.dropna(subset=[plot_col])
        else:
            return None, None
        
        if current_df.empty:
            return None, None
        
        # Create pivot table
        pivot_data = current_df.pivot(
            index=variable_config["iterate_over"][1],  # y-axis
            columns=variable_config["iterate_over"][0],  # x-axis
            values=plot_col
        )
        
        if pivot_data.empty:
            return None, None
            
        # Convert pivot table to float
        pivot_data = pivot_data.astype(float)
        if reverse_x_axis:
            pivot_data = pivot_data.iloc[::-1]
        
        return current_df, pivot_data
    
    # Helper function to create a heatmap on a given axis
    def create_heatmap(ax, data, cmap, vmin, vmax, x_label, y_label, title, colorbar_label=None, show_x_label=True, show_y_label=True, tick_fontsize=10, axis_label_fontsize=12):
        # Create heatmap with colorbar at specified location
        if colorbar_location == 'bottom':
            hm = sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                        annot=annot, fmt=fmt, annot_kws=annot_kws, cbar=True,
                        cbar_kws={'orientation': 'horizontal', 'location': 'bottom'},
                        linewidths=0, linecolor=None)  # Remove cell divider lines
            
            # Customize colorbar
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=tick_fontsize)
            
            # Add colorbar label if provided
            if colorbar_label:
                cbar.ax.set_xlabel(colorbar_label, fontsize=axis_label_fontsize, labelpad=10)
        elif colorbar_location == None:
            # do not show colorbar
            hm = sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                        annot=annot, fmt=fmt, annot_kws=annot_kws, cbar=False,
                        linewidths=0, linecolor=None)  # Remove cell divider lines
        else:  # default to right
            hm = sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                        annot=annot, fmt=fmt, annot_kws=annot_kws, cbar=True,
                        linewidths=0, linecolor=None)  # Remove cell divider lines
            
            # Customize colorbar for right position too
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=tick_fontsize)
            
            # Add colorbar label if provided
            if colorbar_label:
                cbar.ax.set_ylabel(colorbar_label, fontsize=axis_label_fontsize, labelpad=10)
        
        # Reverse axes if requested
        if reverse_x_axis:
            ax.invert_xaxis()
        
        if reverse_y_axis:
            ax.invert_yaxis()
        
        # Set labels and title
        if show_x_label:
            # Use custom x-axis label if provided
            if x_axis_label is not None:
                ax.set_xlabel(x_axis_label, fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
        else:
            ax.set_xlabel('')
        
        if show_y_label:
            # Use custom y-axis label if provided
            if y_axis_label is not None:
                ax.set_ylabel(y_axis_label, fontsize=axis_label_fontsize)
            else:
                ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
        else:
            ax.set_ylabel('')
        
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        
        # Set tick font sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
        return hm
    
    # Convert single value_column to list for uniform processing
    if not isinstance(value_columns, list):
        value_columns = [value_columns]
    num_columns = len(value_columns)
    
    # Normalize parameters to appropriate lists
    titles = standardize_param_list(titles, None, num_columns)
    vmins = standardize_param_list(vmins, None, num_columns)
    vmaxs = standardize_param_list(vmaxs, None, num_columns)
    cmaps = standardize_param_list(cmaps, 'viridis', num_columns)
    
    # Standardize colorbar_labels
    if colorbar_labels is not None:
        if not isinstance(colorbar_labels, list):
            colorbar_labels = [colorbar_labels] * num_columns
        colorbar_labels = colorbar_labels[:num_columns] + [None] * (num_columns - len(colorbar_labels))
    else:
        colorbar_labels = [None] * num_columns
    
    # Process needed columns for callables
    if needed_cols_for_callable is not None:
        if not any(isinstance(item, list) for item in needed_cols_for_callable):
            needed_cols_for_callable = [needed_cols_for_callable] * num_columns
        needed_cols_for_callable = needed_cols_for_callable[:num_columns] + [None] * (num_columns - len(needed_cols_for_callable))
    else:
        needed_cols_for_callable = [None] * num_columns
    
    # Filter DataFrame based on fixed values
    df_filtered = df.copy()
    for col, val in fixed_values.items():
        df_filtered = df_filtered[df_filtered[col] == val]
    
    # Handle case where there's no grid_by variable
    if not variable_config.get("grid_by"):
        num_grid_rows = 1
        
        # Create a single row of subplots
        if ax is None:
            if figsize is None:
                figsize = (6 * num_columns, 4)
            fig, axes = plt.subplots(num_grid_rows, num_columns, figsize=figsize)
            if num_columns == 1:
                axes = np.array([[axes]])  # Make it indexable as 2D array
            else:
                axes = np.array(axes).reshape(1, -1)  # Ensure 2D array
            plt.subplots_adjust(left=0.15)
        else:
            if isinstance(ax, np.ndarray):
                if ax.ndim == 1 and num_columns > 1:
                    axes = ax.reshape(1, -1)  # Convert 1D array to 2D
                else:
                    axes = ax
            else:
                axes = np.array([[ax]])  # Single axis to 2D array
            fig = axes[0, 0].figure
        
        # Process each value column
        for col_idx, value_column in enumerate(value_columns):
            current_ax = axes[0, col_idx]
            current_needed_cols = needed_cols_for_callable[col_idx]
            
            # Prepare data for this column
            _, pivot_data = prepare_data(df_filtered, value_column, current_needed_cols)
            
            if pivot_data is None:
                print(f"No data available for subplot {col_idx+1}")
                continue
            
            # Format axis labels
            x_label = variable_config["iterate_over"][0].replace('_', ' ').title()
            y_label = variable_config["iterate_over"][1].replace('_', ' ').title()
            
            # Create and format heatmap
            create_heatmap(
                current_ax, pivot_data, cmaps[col_idx], vmins[col_idx], vmaxs[col_idx],
                x_label, y_label, titles[col_idx], colorbar_label=colorbar_labels[col_idx],
                show_x_label=True,
                show_y_label=True,
                tick_fontsize=tick_fontsize,
                axis_label_fontsize=axis_label_fontsize
            )
    else:
        # Get unique values for grid_by variable
        grid_values = sorted(df_filtered[variable_config["grid_by"]].unique())
        num_grid_values = len(grid_values)
        
        if num_grid_values == 0:
            raise ValueError(f"No unique values found for grid_by variable '{variable_config['grid_by']}'")
        
        # Determine subplot layout based on grid_by_column
        if grid_by_column:
            # Grid values in columns, value columns in rows
            num_rows, num_cols = num_columns, num_grid_values
        else:
            # Grid values in rows, value columns in columns (original behavior)
            num_rows, num_cols = num_grid_values, num_columns
        
        # Create figure and axes if not provided
        if ax is None:
            if figsize is None:
                # Adjust default figsize based on layout
                if grid_by_column:
                    figsize = (6 * num_grid_values, 4 * num_columns)
                else:
                    figsize = (6 * num_columns, 4 * num_grid_values)
                    
            fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
            
            # Make axes a 2D array for consistent indexing
            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1:
                axes = np.array([axes]).reshape(1, -1)
            elif num_cols == 1:
                axes = np.array(axes).reshape(-1, 1)
                
            plt.subplots_adjust(left=0.15)
        else:
            if isinstance(ax, np.ndarray):
                if ax.ndim == 1:
                    if num_rows == 1:
                        axes = ax.reshape(1, -1)  # Single row
                    else:
                        axes = ax.reshape(-1, 1)  # Single column
                else:
                    axes = ax
            else:
                axes = np.array([[ax]])  # Single axis to 2D array
            fig = axes[0, 0].figure
        
        # Plot heatmaps based on the layout orientation
        for grid_idx, grid_value in enumerate(grid_values):
            grid_df = df_filtered[df_filtered[variable_config["grid_by"]] == grid_value]
            
            for col_idx, value_column in enumerate(value_columns):
                # Determine the current axis based on layout orientation
                if grid_by_column:
                    # Value columns in rows, grid values in columns
                    row_idx, col_idx_adj = col_idx, grid_idx
                else:
                    # Grid values in rows, value columns in columns (original)
                    row_idx, col_idx_adj = grid_idx, col_idx
                
                current_ax = axes[row_idx, col_idx_adj]
                current_needed_cols = needed_cols_for_callable[col_idx]
                
                # Prepare data for this column
                _, pivot_data = prepare_data(grid_df, value_column, current_needed_cols)
                
                if pivot_data is None:
                    print(f"No data available for {variable_config['grid_by']} = {grid_value}, subplot {col_idx+1}")
                    continue
                
                # Format axis labels
                x_label = variable_config["iterate_over"][0].replace('_', ' ').title()
                y_label = variable_config["iterate_over"][1].replace('_', ' ').title()
                
                # Determine which subplot gets title and which gets axis labels
                if grid_by_column:
                    show_title = (row_idx == 0)
                    show_x_label = (row_idx == num_rows-1)
                    show_y_label = (col_idx_adj == 0)
                    title_text = f"{grid_value}" if show_title else None
                else:
                    show_title = (grid_idx == 0)
                    show_x_label = (row_idx == num_rows-1)
                    show_y_label = (col_idx_adj == 0)
                    title_text = titles[col_idx] if show_title else None
                
                # Create and format heatmap
                create_heatmap(
                    current_ax, pivot_data, cmaps[col_idx], vmins[col_idx], vmaxs[col_idx],
                    x_label, y_label, title_text, colorbar_label=colorbar_labels[col_idx],
                    show_x_label=show_x_label,
                    show_y_label=show_y_label,
                    tick_fontsize=tick_fontsize,
                    axis_label_fontsize=axis_label_fontsize
                )
                
                # Add annotations based on layout
                if grid_by_column:
                    # Add column headers (grid values) at the top
                    if row_idx == 0:
                        grid_name = variable_config["grid_by"].replace('_', ' ').title()
                        if variable_config["grid_by"] == "mlp_expansion_factor":
                            grid_name = "MLP Factor"
                        elif variable_config["grid_by"] == "checkpoint":
                            grid_name = "Training Steps"
                        current_ax.set_title(f'{grid_name}: {grid_value}', fontsize=title_fontsize)
                    
                    # Add row labels (value column names) on the left
                    if col_idx_adj == 0:
                        if titles[col_idx] is not None:
                            title_text = titles[col_idx]
                            current_ax.text(-0.4, 0.5, title_text, 
                                        rotation=90, transform=current_ax.transAxes, va='center',
                                        fontsize=axis_label_fontsize)
                else:
                    # Original behavior - add grid value annotations on the left
                    if col_idx_adj == 0:
                        grid_name = variable_config["grid_by"].replace('_', ' ').title()
                        if variable_config["grid_by"] == "mlp_expansion_factor":
                            grid_name = "MLP Factor"
                        elif variable_config["grid_by"] == "checkpoint":
                            grid_name = "Training Steps"
                        current_ax.text(-0.4, 0.5, f'{grid_name}: {grid_value}', 
                                      rotation=90, transform=current_ax.transAxes, va='center',
                                      fontsize=axis_label_fontsize)
    
    # Adjust layout based on colorbar location
    if colorbar_location == 'bottom':
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at the bottom for colorbars
    else:  # default to right
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for colorbars
    
    # Add a bit more space between subplots
    plt.subplots_adjust(wspace=0.4)
    
    return fig, axes

def plot_main_with_sideplots(
    fig, 
    main_ax,
    df, 
    variable_config, 
    fixed_values, 
    value_columns,
    line_colors,
    fixed_x_index,
    fixed_y_index,
    top_left_label="",
    top_right_label="",
    right_top_label="",
    right_bottom_label="",
    side_plot_label_fontsize=15,
    side_plot_tick_fontsize=10,
    is_heatmap=True,
    right_plot_ax_lim_scaling=1.1,
    top_plot_ax_lim_scaling=1.1,
):
    """
    Creates a heatmap with side plots showing slices at fixed x and y values.
    
    Args:
        fig: The figure object
        axes: Dictionary of axes {'main', 'top', 'right'}
        df: DataFrame containing the data
        variable_config: Dictionary with keys 'fixed' and 'iterate_over'
        fixed_values: Dictionary with fixed values for the plot
        value_columns: List of column names, can also be list of lists for different lines in each side plot.
        line_colors: List of colors for the line plots [color1, color2, ...]
        fixed_x_index: Index in the x-axis array to use for the right side plot
        fixed_y_index: Index in the y-axis array to use for the top plot
        top_left_label: Label for the top left y-axis
        top_right_label: Label for the top right y-axis
        right_top_label: Label for the right top x-axis
        right_bottom_label: Label for the right bottom x-axis
        side_plot_label_fontsize: Fontsize for the side plot labels
        side_plot_tick_fontsize: Fontsize for the side plot ticks
        is_heatmap: Whether to add reference lines to the main plot
        right_plot_ax_lim_scaling: Scaling factor for the right plot x-axis limits
        top_plot_ax_lim_scaling: Scaling factor for the top plot y-axis limits
    
    Returns:
        fig: The figure object
        axes: Dictionary of axes {'main', 'top', 'right'}
    """
    # Get the main heatmap axis
    main_position = main_ax.get_position()
    print(main_position)

    # Get unique values for x and y axes
    x_var = variable_config["iterate_over"][0]
    y_var = variable_config["iterate_over"][1]
    
    # Filter dataframe based on fixed values
    filter_query = " & ".join([f"{k} == @fixed_values['{k}']" for k in fixed_values.keys()])
    df_filtered = df.query(filter_query, engine="python")
    
    # Get unique values for x and y axes
    x_values = np.sort(df_filtered[x_var].unique())
    y_values = np.sort(df_filtered[y_var].unique())
    
    # Get the fixed values for the side plots
    fixed_x_value = x_values[fixed_x_index]
    fixed_y_value = y_values[fixed_y_index]

    # Create new axes for the marginal plots - use frameon=True to ensure spines are visible
    top_ax = fig.add_axes([main_position.x0, main_position.y1 + 0.02, 
                          main_position.width, 0.2], frameon=True)
    right_ax = fig.add_axes([main_position.x1 + 0.03, main_position.y0, 
                            0.2, main_position.height], frameon=True)


    # Create top marginal plot (fixed y value)
    top_data = df_filtered[df_filtered[y_var] == fixed_y_value].sort_values(x_var)
    
    # Plot the data directly on the top_ax using indices instead of actual x values
    x_indices = [list(x_values).index(x) for x in top_data[x_var]]
    
    # check if value_columns is a list of lists
    if isinstance(value_columns, list) and isinstance(value_columns[0], list):
        value_columns_top = value_columns[0]
        value_columns_right = value_columns[1]
    else:
        value_columns_top = value_columns
        value_columns_right = value_columns

    for col_idx, value_column in enumerate(value_columns_top):
        top_ax.plot(x_indices, top_data[value_column], 
               color=line_colors[col_idx], linestyle="-", linewidth=2)
    
    # Configure top axis - explicitly set which spines are visible
    top_ax.set_xlim(-0.5, len(x_values)-0.5)  # Match heatmap cell boundaries
    # align it according to the main plot not the x values
    top_ax.set_ylim(top=max(float(top_data[value_column].max()) for value_column in value_columns_top)*top_plot_ax_lim_scaling)
    top_ax.set_ylabel("")  # Remove default y-axis label
    top_ax.set_xticks([])  # Hide x ticks completely
    top_ax.spines['top'].set_visible(False)
    top_ax.spines['right'].set_visible(False)  # Hide right spine
    top_ax.spines['left'].set_visible(True)
    top_ax.spines['bottom'].set_visible(True)
    # Make spines more visible with explicit color and linewidth
    top_ax.spines['left'].set_linewidth(1.0)
    top_ax.spines['bottom'].set_linewidth(1.0)
    top_ax.spines['left'].set_color('black')
    top_ax.spines['bottom'].set_color('black')

    # Add small arrow to top of left y-axis (extremely short stem)
    top_ax.annotate('', xy=(0, 1), xytext=(0, 0.99),
                   xycoords=('axes fraction', 'axes fraction'), 
                   arrowprops=dict(color='black', width=0.02, headwidth=6, headlength=6))
    
    # Add small arrow to right end of bottom x-axis (extremely short stem)
    top_ax.annotate('', xy=(1, 0), xytext=(0.99, 0),
                   xycoords=('axes fraction', 'axes fraction'), 
                   arrowprops=dict(color='black', width=0.02, headwidth=6, headlength=6))

    # Add label above left y-axis arrow
    top_ax.text(0, 1, top_left_label, fontsize=side_plot_label_fontsize, 
               transform=top_ax.transAxes, va='bottom', ha='center')

    # Create right marginal plot (fixed x value)
    right_df = df_filtered[df_filtered[x_var] == fixed_x_value].sort_values(y_var)

    # Use indices for the y-axis
    y_indices = [list(y_values).index(y) for y in right_df[y_var]]

    for col_idx, value_column in enumerate(value_columns_right):
        right_ax.plot(right_df[value_column], 
                     y_indices, 
                     color=line_colors[col_idx], linestyle="-", linewidth=2)

    # Configure right axis - explicitly set which spines are visible
    right_ax.set_ylim(-0.5, len(y_values)-0.5)  # Match heatmap cell boundaries
    right_ax.set_xlim(right=max(float(right_df[value_column].max()) for value_column in value_columns_right)*right_plot_ax_lim_scaling)
    right_ax.set_xlabel(right_bottom_label, fontsize=side_plot_label_fontsize)  # Use the customizable bottom label
    right_ax.set_yticks([])  # Hide y ticks completely
    right_ax.spines['top'].set_visible(False)  # Hide top spine
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['left'].set_visible(True)
    right_ax.spines['bottom'].set_visible(True)
    # Make spines more visible with explicit color and linewidth
    right_ax.spines['left'].set_linewidth(1.0)
    right_ax.spines['bottom'].set_linewidth(1.0)
    right_ax.spines['left'].set_color('black')
    right_ax.spines['bottom'].set_color('black')

    # Add small arrow to top of left y-axis (extremely short stem)
    right_ax.annotate('', xy=(0, 1), xytext=(0, 0.99),
                     xycoords=('axes fraction', 'axes fraction'), 
                     arrowprops=dict(color='black', width=0.02, headwidth=6, headlength=6))

    # Add small arrow to right end of bottom x-axis (extremely short stem)
    right_ax.annotate('', xy=(1, 0), xytext=(0.99, 0),
                     xycoords=('axes fraction', 'axes fraction'), 
                     arrowprops=dict(color='black', width=0.02, headwidth=6, headlength=6))

    # Add label to top x-axis
    right_ax.text(0.5, 1.05, right_top_label, fontsize=side_plot_label_fontsize,
                 transform=right_ax.transAxes, va='center', ha='center')

    # Find indices for the fixed values
    y_idx = list(y_values).index(fixed_y_value)
    x_idx = list(x_values).index(fixed_x_value)

    # Add reference lines to the main plot
    if is_heatmap:
        main_ax.axhline(y=y_idx+0.5, color='black', linestyle='--', alpha=1, linewidth=1)
        main_ax.axvline(x=x_idx+0.5, color='black', linestyle='--', alpha=1, linewidth=1)
    else:
        main_ax.axhline(y=fixed_y_value, color='black', linestyle='--', alpha=1, linewidth=1)
        main_ax.axvline(x=fixed_x_value, color='black', linestyle='--', alpha=1, linewidth=1)

    # Add reference line to the top marginal plot (vertical line at fixed x value)
    top_ax.axvline(x=x_idx, color='black', linestyle='--', alpha=1, linewidth=1)

    # Add reference line to the right marginal plot (horizontal line at fixed y value)
    right_ax.axhline(y=y_idx, color='black', linestyle='--', alpha=1, linewidth=1)
    
    # Set tick fontsize for top and right plots
    top_ax.tick_params(axis='y', labelsize=side_plot_tick_fontsize)
    right_ax.tick_params(axis='x', labelsize=side_plot_tick_fontsize)

    return fig, {'main': main_ax, 'top': top_ax, 'right': right_ax}


def plot_vector_field(
    transformer_df, 
    fixed_values,
    figsize=(10, 8),
    scale=30,               # Overall vector scaling (smaller = larger vectors)
    arrow_width=0.005,      # Width of arrow shaft
    arrow_headwidth=7,      # Width of arrow head
    arrow_headlength=8,     # Length of arrow head
    blue_color=None, 
    red_color=None, 
    sqrt_scale_x_axis=True,
    x_min=None,
    x_max=None,
    y_min=2,
    y_max=12,
    background_cmap="coolwarm",
):
    # Filter the dataframe
    filtered_df = transformer_df[
        (transformer_df['num_dims'] == fixed_values["num_dims"]) & 
        (transformer_df['context_length'] == fixed_values["context_length"]) & 
        (transformer_df['mlp_expansion_factor'] == fixed_values["mlp_expansion_factor"])
    ]
    
    if filtered_df.empty:
        print(f"No data found for the specified parameters")
        return None, None
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Extract unique checkpoints and num_tasks
    checkpoints = sorted(filtered_df['checkpoint'].unique())
    num_tasks_values = sorted(filtered_df['num_tasks'].unique())
    
    # Reduce density - only use every 3rd checkpoint but keep all task values
    checkpoints = checkpoints[::3]
    
    # Create meshgrid for vector field
    X, Y = np.meshgrid(checkpoints, num_tasks_values)
    
    # Initialize arrays for vector components
    U = np.zeros_like(X, dtype=float)  # x-component (derivative over N)
    V = np.zeros_like(Y, dtype=float)  # y-component (derivative over D)
    
    # Fill in vector components
    for i, num_tasks in enumerate(num_tasks_values):
        for j, checkpoint in enumerate(checkpoints):
            # Find the corresponding row
            row = filtered_df[
                (filtered_df['num_tasks'] == num_tasks) & 
                (filtered_df['checkpoint'] == checkpoint)
            ]
            
            if not row.empty:
                # Get the derivatives using the correct column names
                U[i, j] = row['relative_distance_train_second_derivative_over_N'].values[0]
                V[i, j] = row['relative_distance_train_second_derivative_over_D'].values[0]
    
    # FLIP X DIRECTION: Negate U values to flip horizontal direction - the direction of second derivative over N should point left when positive
    U = -U
    
    # Get original sign and magnitude
    U_sign = np.sign(U)
    V_sign = np.sign(V)
    U_mag = np.abs(U)
    V_mag = np.abs(V)
    
    # Find the 90th percentile of the magnitudes to determine scaling
    u_90p = np.nanpercentile(U_mag[U_mag > 0], 90) if np.any(U_mag > 0) else 1.0
    v_90p = np.nanpercentile(V_mag[V_mag > 0], 90) if np.any(V_mag > 0) else 1.0
    # create normalizing factor as a function of the 90th percentile of the magnitudes
    normalizing_factor = np.max([u_90p, v_90p])
    
    # Normalize
    U_norm = U_sign * U_mag / normalizing_factor  # Normalize U
    V_norm = V_sign * V_mag / normalizing_factor  # Normalize V
    
    # Calculate color based on direction of vectors
    colors = np.empty(U_norm.flatten().shape, dtype=object)
    
    # Convert hex colors to RGBA 
    red_rgba = np.array(to_rgba(red_color))
    blue_rgba = np.array(to_rgba(blue_color))
    
    # White color for interpolation
    white_rgba = np.array([1, 1, 1, 1])
    
    for idx in range(len(colors)):
        u_val = U_norm.flatten()[idx]
        v_val = V_norm.flatten()[idx]
        
        # Choose base color based on sign of V
        base_color = blue_rgba if v_val > 0 else red_rgba
        
        # Calculate weakening factor based on ratio of |U| to |V| (if U and V are NOT in opposite directions)
        u_mag = abs(u_val)
        v_mag = abs(v_val)

        if u_val * v_val > 0:
            # Avoid division by zero
            if v_mag < 1e-6:
                weakening_factor = 1.0  # Maximum weakening
            else:
                # Calculate ratio of |U| to |V|
                ratio = u_mag / v_mag
                # Convert to a weakening factor (0 = no weakening, 1 = full weakening)
                weakening_factor = min(ratio / 3, 1.0)  # Adjust the divisor to control weakening rate
            
            # Interpolate between base color and white
            color_rgba = (1 - weakening_factor) * base_color + weakening_factor * white_rgba
        else:
            color_rgba = base_color
        
        # Convert to hex
        colors[idx] = to_hex(color_rgba)


    # First, plot a background heatmap showing the relative distance
    rel_dist = np.zeros_like(X, dtype=float)
    for i, num_tasks in enumerate(num_tasks_values):
        for j, checkpoint in enumerate(checkpoints):
            row = filtered_df[
                (filtered_df['num_tasks'] == num_tasks) & 
                (filtered_df['checkpoint'] == checkpoint)
            ]
            if not row.empty:
                rel_dist[i, j] = row['relative_distance_train'].values[0]
    
    # Use pcolormesh to create a heatmap background
    heatmap = ax.pcolormesh(X, Y, rel_dist, cmap=background_cmap, alpha=0.5, shading='auto')
    
    # Plot vector field with customizable appearance
    q = ax.quiver(X, Y, U_norm, V_norm, color=colors.flatten(), 
                 scale=scale, width=arrow_width, 
                 headwidth=arrow_headwidth, headlength=arrow_headlength)
    
    # Set labels with better formatting
    ax.set_xlabel('Training Steps (N)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Task Diversity (D)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Use log scale for y-axis (Task Diversity)
    ax.set_yscale('log', base=2)
    # Use sqrt scale for x-axis if requested
    if sqrt_scale_x_axis:
        # Create a custom sqrt scale function
        from matplotlib.scale import ScaleBase, register_scale
        from matplotlib.transforms import Transform
        
        class SqrtScale(ScaleBase):
            name = 'sqrt'
            
            def get_transform(self):
                return SqrtTransform()
            
            def set_default_locators_and_formatters(self, axis):
                pass
        
        class SqrtTransform(Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True
            
            def transform_non_affine(self, a):
                return np.sqrt(a)
            
            def inverted(self):
                return SquaredTransform()
        
        class SquaredTransform(Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True
            
            def transform_non_affine(self, a):
                return a * a
            
            def inverted(self):
                return SqrtTransform()
        
        register_scale(SqrtScale)
        ax.set_xscale('sqrt')
        
        # Set x-axis limits if provided, otherwise use data range with small padding
        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
            print(f"X-axis limits: {x_min} to {x_max}")
        else:
            # Default padding (5% of range)
            min_checkpoint = min(checkpoints)
            max_checkpoint = max(checkpoints)
            x_range = max_checkpoint - min_checkpoint
            padding_x = x_range * 0.05
            
            # Set x-axis limits with padding
            min_limit = max(0, min_checkpoint - padding_x)
            max_limit = max_checkpoint + padding_x
            ax.set_xlim(min_limit, max_limit)
            print(f"X-axis limits (auto): {min_limit} to {max_limit}")
                
        ax.set_xticks(checkpoints)
        ax.set_xticklabels([f"{int(x)}" for x in checkpoints], rotation=90)
    
    # Set y-axis limits with explicit values in powers of 2
    ax.set_ylim(2**y_min, 2**y_max)
    print(f"Y-axis limits: {2**y_min} to {2**y_max}")

    # Add these two lines here:
    # Explicitly set y-ticks to show all task diversity values
    ax.set_yticks(num_tasks_values)

    # Ensure only even y-ticks are visible by adjusting tick parameters
    ax.tick_params(axis='y', which='both', length=4, width=1, pad=10, labelsize=10)
    # remove all odd y-ticks
    ax.set_yticks(num_tasks_values[::2])
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', color='#cccccc', which='both')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    
    plt.tight_layout()
    return fig, ax