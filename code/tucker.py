import time, os
import numpy as np

from pyttb.tensor import tensor as ttb
from pyttb.tucker_als import tucker_als
from pyttb.hosvd import hosvd
from util import empty_the_dir
from video_handler import save_numpy_as_video

def calculate_tucker_compression_ratio(ranks, shape, print_ = True):
    """
    Calculates theoretical compression ratio of using Tucker format given shape of full tensor and ranks(aka shape of the core)
    """
    if len(ranks) != len(shape):
        raise ValueError("ranks and shape must be of the same length")
    rank_product = 1 # core memory size
    shape_product = 1 # full dense tensor memory size
    unfoldings_memory_size = 0
    for i in range(len(ranks)):
        rank_product *= ranks[i]
        shape_product *= shape[i]
        unfoldings_memory_size += ranks[i]*shape[i]
    c_r = shape_product / (rank_product + unfoldings_memory_size)
    if print_:
        print(f"Compression ratio is about{c_r: .3f}, so Tucker format requires approximately{100/c_r: .2f}% of dense tensor memory size")
    return c_r

def calculate_suggestion_tucker_ranks(desired_comp_ratio, shape, print_ = True):
    """
    Calculates ranks for desired compression ratio, finding appropriate constant C such that C * shape = ranks
    """
    if shape[-1] == 3:
        c = desired_comp_ratio**(-1/len(shape[:-1]))
        ranks = [(c*r).__floor__() for r in shape[:-1]] + [3]
    else:
        c = desired_comp_ratio**(-1/len(shape))
        ranks = [(c*r).__floor__() for r in shape]
    while calculate_tucker_compression_ratio(ranks, shape, print_ = False) < desired_comp_ratio:
        ranks[np.argmax(ranks)] -= 1
    if print_:
        print(f"Suggested ranks for desired compression ratio {desired_comp_ratio} is {ranks}, exact comression ratio is {calculate_tucker_compression_ratio(ranks, shape, print_ = False): .3f}")
    return ranks 



def decomposition_tucker(exp_name,
                         np_tensor,
                         method,
                         method_keyargs,
                         desired_comp_ratio=None,
                         ranks = None,
                         mean_frame_substraction = False, 
                         fps=20, 
                         format_="mp4v",
                         colorful=True, 
                         delete_prev=False):
    """ 
    Tucker format decomposition and reconstruction, based on pyttb for details not described see pyttb docs of pytbb.tucker_als and pytbb.hosvd
    np_tensor: numpy.ndarray - tensor to decompose
    init: str - either 'random' or 'nvecs' see pyttb docs of tucker_als 
    desired_comp_ratio: float - if specified and ranks are not given, then it's used to calculate ranks
    mean_frame_substraction: bool - if True substracts mean from each frame before decomposing, adds it back aftewards

    returns:
    results_dict: dictionary
        "Tucker_reconstruction": ttb.ttensor - Tucker formated reconstruction (core and matrices) potentially mean centered, so if mean_rame_substraction was applied, then it's not equivalent to np_tensor
        "dense_reconstruction": ttb.tensor - full reconstruction tensor (always equivalent to np_tensor) 
        "is_same_tensor": bool - True if T and result are same tensor in different formats (when mean frame was not substracted)
        "relative_error": float - || np_tensor - dense_reconstruction ||_F / ||np_tensor||_F
        "method_time": float - time required for tensor method
        "mean_centering_extra_time": float - time required for mean substraction before method and addition after for reconstruction
    """
    total_start = time.time()
    
    if desired_comp_ratio and ranks:
        raise ValueError("Too many param values are specified, either 'desired_comp_ratio' or 'ranks' must be not None")
    elif not (desired_comp_ratio or ranks):
        raise ValueError("Too few param values are specified, either 'desired_comp_ratio' or 'ranks' must be not None")


    output_dir = f"../videos/output/{method}/{exp_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif delete_prev:
        empty_the_dir(output_dir)
        print(f"Removed previous results from {output_dir}!!!")
    else:
        raise ValueError(f"Experiment with such name '{exp_name}' already exists, delete folder {output_dir} or pass delete_prev = True.")
        
    mean_frame = None
    ttb_tensor = ttb.from_data(np_tensor)
    if mean_frame_substraction:
        start = time.time()
        mean_frame = np.mean(np_tensor, axis=0)
        np_tensor_minus_mean = np.array([frame - mean_frame for frame in np_tensor])
        mean_substraction_time = time.time() - start
        print(f"Mean substraction took {mean_substraction_time: .2f} seconds.")
        ttb_tensor = ttb.from_data(np_tensor_minus_mean)
    
    if desired_comp_ratio:
        ranks = calculate_suggestion_tucker_ranks(desired_comp_ratio, ttb_tensor.shape)                                              
    else:
        calculate_tucker_compression_ratio(ranks, ttb_tensor.shape)
    
    start = time.time()
    tol = 1e-3
    if "tol" in method_keyargs.keys():
        tol = method_keyargs["tol"]
    if method == "TuckerALS":
        init = "random"
        if "init" in method_keyargs.keys():
            if method_keyargs["init"] == "nvecs":
                init = "nvecs"
            
        T, T_init, output_dict = tucker_als(ttb_tensor,rank=ranks, stoptol=tol, init=init)
    elif method == "HOSVD":
        verbosity = 3
        if "verbosity" in method_keyargs.keys():
            verbosity = int(method_keyargs["verbosity"])
    
        T = hosvd(ttb_tensor, tol=tol, verbosity=verbosity, ranks=ranks)
    else:
        raise ValueError(f"Wrong value of method, got '{method}', but allowed are only: 'HOSVD' and 'TuckerALS'")
    method_time = time.time() - start
    print(f"{method} took {method_time: .2f} seconds.")

    result = T.full()    
        
    if mean_frame_substraction:
        start = time.time()
        result.data = np.array([frame + mean_frame for frame in result.data])
        mean_addition_time = time.time() - start
        print(f"Mean addition back took {mean_addition_time: .2f} seconds.")

    relative_error = np.linalg.norm(np_tensor - result.data) / np.linalg.norm(np_tensor)
    print(f"(Frobenius) || original - reconstructed || / ||original|| = {relative_error: .3f}")
    
    save_numpy_as_video(result.data,
                        f"{output_dir}/{method}_{exp_name}_ranks_{'_'.join(str(r) for r in ranks)}_{'colorful' if colorful else 'gray'}{'_minus_subtracted' if mean_frame_substraction else ''}.mp4",
                        fps=fps,
                        format_=format_,
                        colorful=colorful)

    print(f"Total: {time.time() - total_start: .2f} seconds.")
    is_same_tensor = not mean_frame_substraction
    results_dict = dict()
    results_dict["Tucker_reconstruction"] = T
    results_dict["dense_reconstruction"] = result
    results_dict["is_same_tensor"] = is_same_tensor
    results_dict["relative_error"] = relative_error
    results_dict["method_time"] = method_time
    if mean_frame_substraction:
        results_dict["mean_centering_extra_time"] = mean_addition_time + mean_substraction_time
    return results_dict

    


    