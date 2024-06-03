import os, time
import numpy as np

from tqdm.auto import tqdm
from util import empty_the_dir
from video_handler import save_numpy_as_video

def rSVD(X, r, p=0, q=0):
    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    # only when q is not 0
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z,mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q @ UY

    return U, S, VT


def t_slice(tensor, counter, dim):
    if dim > 2:
        raise ValueError("slice along dim > 2 is not implemented, since colorfull video is 4-th order tensor is most high-dim case we consider.")
    if dim == 0:
        return tensor[counter]
    elif dim == 1:
        return tensor[:, counter, ...]
    else:
        return tensor[:, :, counter, ...]
        
def slice_by_slice_RSVD(tensor, ranks, colorful, dim=0, reconstruct=True):
    # dim = 0 <-> frame by frame, other values -> other slices
    lefts = [] # U matrices
    sigmas = [] #
    rights = [] # (V.T) matrices
    reconstructed = np.zeros(shape=tensor.shape, dtype=np.uint8)
    # ranks are either specified for each slice or given int are comupted equally for all slices
    if type(ranks) == int:
        ranks = [ranks] * tensor.shape[dim]
    if len(ranks) != tensor.shape[dim]:
        raise ValueError(f"Wrong ranks, expected a list of integers with length = number of frames, here {tensor.shape[0]} frames, or an integer, but got {ranks}")
    if not colorful:
        for frame_counter in tqdm(range(tensor.shape[dim]), desc=f"RSVD of each slice"):
            r = ranks[frame_counter]
            rU, rS, rVT = rSVD(t_slice(tensor, frame_counter, dim),r)
            lefts.append(rU)
            sigmas.append(rS)
            rights.append(rVT)
            if reconstruct:
                if dim == 0:
                    reconstructed[frame_counter] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
                elif dim == 1:
                    reconstructed[:, frame_counter, ...] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
                elif dim == 2:
                    reconstructed[:, :, frame_counter, ...] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
    else:
        for frame_counter in tqdm(range(tensor.shape[dim]), desc=f"RSVD of each slice"):
            r = ranks[frame_counter]
            frame_sigmas = []
            channels_reconstructed = []
            for color_channel in range(tensor.shape[-1]):
                rU, rS, rVT = rSVD(t_slice(tensor, frame_counter, dim)[..., color_channel],r)

                if dim == 0:
                    reconstructed[frame_counter,..., color_channel] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
                elif dim == 1:
                    reconstructed[:, frame_counter, :, color_channel] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
                elif dim == 2:
                    reconstructed[:, :, frame_counter, color_channel] = np.array(rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:])
                    
                frame_sigmas.append(rS)
            sigmas.append(np.array(frame_sigmas))

    
    if reconstruct:
        relative_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
        print(f"(Frobenius) || original - reconstructed || / ||original|| = {relative_error: .3f}")
        return lefts, sigmas, rights, reconstructed, relative_error
    else:
        return lefts, sigmas, rights

def all_dim_RSVD(exp_name, video_tensor, desired_compression_ratio, format_='mp4v', fps=20, colorful=True, delete_prev=False, save=True):
    tensors = []
    sigmas = []
    relative_errors = []
    ranks = calculate_RSVD_ranks(desired_compression_ratio, video_tensor.shape)
    exact_comp_ratios = calculate_RSVD_compression_ratios(ranks, video_tensor.shape)
    times = []
    output_dir = f"../videos/output/RSVD/{exp_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif delete_prev:
        empty_the_dir(output_dir)
        print(f"Removed previous results from {output_dir}!!!")
    else:
        raise ValueError(f"Experiment with such name '{exp_name}' already exists, delete folder {output_dir} or pass delete_prev = True.")
        
    for i in range(3):
        print(f"RSVD slice by slice along {i}-th dimension with rank = {ranks[i]}:")
        
        start = time.time()
        left, sigma, right, reconstruction, relative_error =  slice_by_slice_RSVD(video_tensor, ranks=ranks[i], dim=i, colorful=colorful)
        times.append(time.time() - start)
        tensors.append(reconstruction)
        sigmas.append(sigma)
        relative_errors.append(relative_error)
        if save:
            save_numpy_as_video(reconstruction,
                                f"{output_dir}/RSVD_{exp_name}_dim_{i}_rank_{ranks[i]}_{'colorful' if colorful else 'gray'}.mp4",
                                fps=fps,
                                format_=format_,
                                colorful=colorful)
        print("-"*100)
    results_dict = dict()
    results_dict["reconstructed_tensors"] = tensors
    results_dict["sigmas"] = sigmas
    results_dict["relative_errors"] = relative_errors
    results_dict["ranks"] = ranks
    results_dict["exact_comp_ratios"] = exact_comp_ratios
    results_dict["times"] = times
    return results_dict


def calculate_RSVD_ranks(desired_comp_ratio, shape):
    ranks = [0, 0, 0]
    shape = shape[:3]
    shape_sum = sum(shape)
    desired_total_size = 1
    for s in shape:
        desired_total_size *= s
    desired_total_size /= desired_comp_ratio
    for i in range(3):
        ranks[i]  = int((desired_total_size / (shape[i] * (shape_sum - shape[i]))).__floor__())
    return ranks
    
def calculate_RSVD_compression_ratios(ranks, shape):
    comp_ratios = [0, 0, 0]
    shape = shape[:3]
    total_size = 1
    for s in shape:
        total_size *= s
    for i in range(3):
        comp_ratios[i] = total_size / (ranks[i] * shape[i] * (sum(shape) - shape[i]))
    return comp_ratios
