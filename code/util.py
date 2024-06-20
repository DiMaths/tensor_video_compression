import os, shutil, time, json
import pyttb.tensor as ttb
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def empty_the_dir(dir_path: str) -> None:
    with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)

def load_ttb_from_npy_file(path_to_npy: str) -> None:
    with open(path_to_npy, 'rb') as f:
        tensor = np.load(f)
    return ttb.from_data(tensor)


def decomposition_results_to_json(exp_name: str, method: str, results_dict: dict) -> None:
    # dump only serializable parts (ignore tensors)
    json_res = dict()
    for key in results_dict.keys():
        json_res[key] = dict()
        for non_series_key in ['relative_error', 'method_time', 'exact_compress_ratio']:
            json_res[key][non_series_key] = results_dict[key][non_series_key]
    json_object = json.dumps(json_res, indent=4) 
    
    with open(f"../jsons/{method}_{exp_name}.json", "w") as outfile:
        outfile.write(json_object)
        print(f"Saved to ../jsons/{method}_{exp_name}.json")


def compute_alpha(err, t, c=np.exp(1)):
    return -np.log(err) / np.log(c + np.array(t))



def mean_center_along_axes(X: np.ndarray, axes=None):
    """
    if axes is None:
        # Compute the mean over the entire array if no axes are provided
        mean = np.mean(X)
        X_centered = X - mean
        return X_centered, mean
    
    # Convert axes to a tuple if it is a single integer
    if isinstance(axes, int):
        axes = (axes,)
    """
    
    start = time.time()    
    mean = np.mean(X, axis=axes, keepdims=True)
    X_centered = X - mean
    t = time.time() - start
    relative_error = np.linalg.norm(X_centered) / np.linalg.norm(X)
    compress_ratio = np.prod([dim for i, dim in enumerate(X.shape) if i in axes])
    return X_centered, t, relative_error, compress_ratio

def get_all_axis_combinations(ndim: int) -> list:
    """Generate all combinations of axes for an array of given number of dimensions."""
    all_combinations = []
    for r in range(1, ndim+1):
        all_combinations.extend(combinations(range(ndim), r))
    return all_combinations


def prepare_mean_reconstruction_measurements_for_plot(X: np.ndarray):
    all_combinations = get_all_axis_combinations(X.ndim)
    relative_errors_mean_substraction = []
    times_mean_substraction = []
    compress_ratios_mean_substraction = []
    for axes in all_combinations:
        _, t, err, c_r = mean_center_along_axes(X, axes=axes)
        relative_errors_mean_substraction.append(err)
        times_mean_substraction.append(t)
        compress_ratios_mean_substraction.append(c_r)
        
    efficiency_mean_substraction = compute_alpha(err=relative_errors_mean_substraction, t=times_mean_substraction)
    result_dict = {'relative_errors': relative_errors_mean_substraction, 
                   'times': times_mean_substraction,
                   'compress_ratios': compress_ratios_mean_substraction,
                   'alphas':  efficiency_mean_substraction,
                   'labels': [f"Mean-{axes}" if axes else "Mean-Frame" for axes in all_combinations]
                   }
    return result_dict
