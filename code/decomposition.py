import time, os
import numpy as np
import tensorly as tl
import pyttb as ttb

from util import empty_the_dir
from video_handler import save_numpy_as_video
from hosvd import hosvd


def calculate_cp_compress_ratio(rank, shape, print_ = True):
    """
    Calculates theoretical compression ratio of using CP format given shape of full tensor and rank(aka number of components)
    """
    c_r = np.prod(shape) / (np.sum(shape) * rank)
    if print_:
        print(f"Compression ratio is about{c_r: .3f}, so CP format requires approximately{100/c_r: .2f}% of dense tensor memory size")
    return c_r

def calculate_suggestion_cp_rank(desired_compress_ratio, shape, print_ = True):
    """
    Calculates CP rank for desired compression ratio
    """
    rank = (np.prod(shape) / (np.sum(shape) * desired_compress_ratio)).__floor__()
    exact_compress_ratio = calculate_cp_compress_ratio(rank, shape, print_ = False)
    if print_:
        print(f"Suggested rank for desired compression ratio {desired_compress_ratio} is {rank}, exact comression ratio is {exact_compress_ratio: .3f}")
    return rank, exact_compress_ratio


def calculate_tucker_compress_ratio(ranks, shape, print_ = True):
    """
    Calculates theoretical compression ratio of using Tucker format given shape of full tensor and ranks(aka shape of the core)
    """
    if len(ranks) != len(shape):
        raise ValueError("ranks and shape must be of the same length")
    
    unfoldings_memory_size = np.sum(np.multiply(ranks, shape))
    c_r = np.prod(shape) / (np.prod(ranks) + unfoldings_memory_size)
    if print_:
        print(f"Compression ratio is about{c_r: .3f}, so Tucker format requires approximately{100/c_r: .2f}% of dense tensor memory size")
    return c_r

def calculate_suggestion_tucker_ranks(desired_compress_ratio, shape, print_ = True):
    """
    Calculates ranks for desired compression ratio, finding appropriate constant C such that C * shape = ranks
    """
    if shape[-1] == 3:
        c = desired_compress_ratio**(-1/len(shape[:-1]))
        ranks = [(c*r).__floor__() for r in shape[:-1]] + [3]
    else:
        c = desired_compress_ratio**(-1/len(shape))
        ranks = [(c*r).__floor__() for r in shape]
    while calculate_tucker_compress_ratio(ranks, shape, print_ = False) < desired_compress_ratio:
        ranks[np.argmax(ranks)] -= 1
    exact_compress_ratio = calculate_tucker_compress_ratio(ranks, shape, print_ = False)
    if print_:
        print(f"Suggested ranks for desired compression ratio {desired_compress_ratio} is {ranks}, exact comression ratio is {exact_compress_ratio: .3f}")
    return ranks, exact_compress_ratio


def decomposition(exp_name,
                 np_tensor,
                 method,
                 method_keyargs,
                 desired_compress_ratio=None,
                 ranks = None,
                 mean_frame_substraction = False, 
                 fps=20, 
                 format_="mp4v",
                 colorful=True, 
                 delete_prev=False):

    total_start = time.time()
    
    if desired_compress_ratio and ranks:
        raise ValueError("Too many param values are specified, either 'desired_compress_ratio' or 'ranks' must be not None")
    elif not (desired_compress_ratio or ranks):
        raise ValueError("Too few param values are specified, either 'desired_compress_ratio' or 'ranks' must be not None")


    output_dir = f"../videos/output/{method}/{exp_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif delete_prev:
        empty_the_dir(output_dir)
        print(f"Removed previous results from {output_dir}!!!")
    else:
        raise ValueError(f"Experiment with such name '{exp_name}' already exists, delete folder {output_dir} or pass delete_prev = True.")
        
    mean_frame = None
    
    if mean_frame_substraction:
        start = time.time()
        mean_frame = np.mean(np_tensor, axis=0)
        np_tensor_minus_mean = np.array([frame - mean_frame for frame in np_tensor])
        mean_substraction_time = time.time() - start
        print(f"Mean substraction took {mean_substraction_time: .2f} seconds.")
        tl_tensor = tl.tensor(np_tensor_minus_mean)
    
    tl_tensor = tl.tensor(np_tensor)
    
    if desired_compress_ratio:
        if method == "CP":
            ranks, exact_compress_ratio = calculate_suggestion_cp_rank(desired_compress_ratio, tl_tensor.shape)      
        else:
            ranks, exact_compress_ratio = calculate_suggestion_tucker_ranks(desired_compress_ratio, tl_tensor.shape)                                              
    else:
        if method == "CP":
            exact_compress_ratio = calculate_cp_compress_ratio(rank, tl_tensor.shape)                                    
        else:
            exact_compress_ratio = calculate_tucker_compress_ratio(ranks, tl_tensor.shape)    
    
    start = time.time()
    tol = 1e-9 if method=="CP" else 1e-3
    
    if "tol" in method_keyargs.keys():
        tol = float(method_keyargs["tol"])
        
    n_iter_max = 100
    if "n_iter_max" in method_keyargs.keys():
        n_iter_max = int(method_keyargs["n_iter_max"])
            
    if method == "Tucker":
        decomposed, rec_errors = tl.decomposition.tucker(tl_tensor, 
                                                         ranks, 
                                                         n_iter_max=n_iter_max, 
                                                         init='random', 
                                                         svd='random_svd', 
                                                         tol=tol, 
                                                         random_state=42, 
                                                         verbose=False, 
                                                         return_errors=True)
    elif method == "NonNegTucker":
        algorithm_name = 'active_set'
        if "algorithm_name" in method_keyargs.keys():
            algorithm_name = method_keyargs["algorithm_name"]
        decomposed, rec_errors = tl.decomposition.non_negative_tucker_hals(tl_tensor, 
                                                                           ranks, 
                                                                           algorithm=algorithm_name, 
                                                                           return_errors=True)
    elif method == "HOSVD":
        decomposed, rec_errors = hosvd(ttb.tensor().from_data(np_tensor), ranks)
        
    elif method == "CP":
        n_samples = np.max(tl_tensor.shape)
        if "n_samples" in method_keyargs.keys():
            n_samples = int(method_keyargs["n_samples"])
        decomposed, rec_errors = tl.decomposition.randomised_parafac(tl_tensor, 
                                                                     ranks, 
                                                                     n_samples=n_samples, 
                                                                     n_iter_max=n_iter_max, 
                                                                     return_errors=True)
    else:
        raise ValueError(f"Wrong value of method, got '{method}', but allowed are only: 'Tucker', 'CP' and 'NonNegTucker'")
    method_time = time.time() - start
    print(f"{method} took {method_time: .2f} seconds.")

    if method == "CP":
        approximation= tl.cp_to_tensor(decomposed)
    elif method == "HOSVD":
        approximation= decomposed.full().data
        rec_errors = [rec_errors]
    else:
        approximation= tl.tucker_to_tensor(decomposed)
        
    if mean_frame_substraction:
        start = time.time()
        approximation= np.array([frame + mean_frame for frame in approximation])
        mean_addition_time = time.time() - start
        print(f"Mean addition back took {mean_addition_time: .2f} seconds.")

    relative_error = rec_errors[-1]
    print(f"Converged in {len(rec_errors)} iterations with relative error of {relative_error: .3f}")
    
    if isinstance(ranks, int):
        ranks_str = str(ranks)
    else:
        ranks_str = '_'.join(str(r) for r in ranks)
    mean_str = '_minus_subtracted' if mean_frame_substraction else ''
    save_numpy_as_video(approximation,
                        f"{output_dir}/{method}_{exp_name}_ranks_{ranks_str}_{'colorful' if colorful else 'gray'}{mean_str}.mp4",
                        fps=fps,
                        format_=format_,
                        colorful=colorful)
    
    
    print(f"Total: {time.time() - total_start: .2f} seconds.")
    is_same_tensor = not mean_frame_substraction
    results_dict = dict()
    
    results_dict["decomposed"] = decomposed
    results_dict["approximation"] = approximation
    results_dict["is_same_tensor"] = is_same_tensor
    results_dict["relative_error"] = relative_error
    results_dict["method_time"] = method_time
    results_dict["exact_compress_ratio"] = exact_compress_ratio
    if mean_frame_substraction:
        results_dict["mean_centering_extra_time"] = mean_addition_time + mean_substraction_time
    return results_dict

    


    