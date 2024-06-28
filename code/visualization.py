import matplotlib.pyplot as plt
import numpy as np
import time
from adjustText import adjust_text
from matplotlib.lines import Line2D

from rsvd import calculate_2d_RSVD_compress_ratio
from util import compute_alpha, prepare_mean_reconstruction_measurements_for_plot

    
def error_time_analysis(exp_name: str, X:np.ndarray, all_results: dict, title:str = None) -> None:
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 18))
    num_labels = 0
    for method in all_results:
        if method != "RSVD":
            num_labels += 1
            if method == "Tucker":
                color = "blue"
            elif method == "HOSVD":
                color = "gray"
            elif method == "CP":
                color = "magenta"
            relative_errors = []
            times = []
            exact_compress_ratios = []
            for j, compress_ratio in enumerate(all_results[method].keys()):
                relative_errors.append(all_results[method][compress_ratio]["relative_error"])
                exact_compress_ratios.append(all_results[method][compress_ratio]["exact_compress_ratio"])
                times.append(all_results[method][compress_ratio]["method_time"])
                
        
            relative_errors = np.array(relative_errors)
            exact_compress_ratios = np.array(exact_compress_ratios)
            times = np.array(times)
            efficiency = compute_alpha(err=relative_errors, t=times)
            
            axs[0].plot(exact_compress_ratios, relative_errors, marker='o', label=method, color=color)
            axs[1].plot(exact_compress_ratios, times, marker='o', label=method, color=color)
            axs[2].plot(exact_compress_ratios, efficiency, marker='o', label=method, color=color)
            
        else:
            num_labels += 3
            results_rsvd = all_results[method]
            relative_errors = np.array([results_rsvd[compress_ratio]["relative_error"] for compress_ratio in results_rsvd.keys()])
            times = np.array([results_rsvd[key]["method_time"] for key in results_rsvd.keys()])
            efficiency = np.array([compute_alpha(err=relative_errors[:, i], t=np.array(times[:, i])) for i in range(3)])
            
            exact_compress_ratios = np.array([results_rsvd[compress_ratio]["exact_compress_ratio"] for compress_ratio in results_rsvd.keys()])
            colors = ["black", "yellow", "green"]
            for i in range(3):
                axs[0].plot(exact_compress_ratios[:, i], relative_errors[:, i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                axs[1].plot(exact_compress_ratios[:, i], times[:, i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                axs[2].plot(exact_compress_ratios[:, i], efficiency[i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                
    axs[-1].set_xlabel("Compression Ratio")
    axs[0].set_title("Relative error of the reconstruction")
    axs[0].set_xscale("log") # for better visual interpretability: log scale?
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
    
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    
    axs[2].set_title("Efficiency")
    axs[2].set_ylabel("alpha = ln(1/error) / ln(e+time)")
    
    all_means_dict = prepare_mean_reconstruction_measurements_for_plot(X)
    
    for i, label in enumerate(all_means_dict['labels']):
        axs[0].scatter(all_means_dict['compress_ratios'][i], all_means_dict['relative_errors'][i], marker='D', color='pink', s=100)
        axs[1].scatter(all_means_dict['compress_ratios'][i], all_means_dict['times'][i], marker='D', color='pink', s=100)
        axs[2].scatter(all_means_dict['compress_ratios'][i], all_means_dict['alphas'][i], marker='D', color='pink', s=100)
    
    max_mean_error = np.max(all_means_dict['relative_errors'])    
    axs[0].set_ylim(-0.05 * max_mean_error, 1.05 * max_mean_error)
    
    # add manual symbols to auto legend
    handles, labels = axs[0].get_legend_handles_labels()
    point = Line2D([0], [0], label='M-(axes) - mean along axes', marker='D', color='pink', markersize=10, linestyle='')
    handles.extend([point])
    axs[0].legend(handles=handles, loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.3))
    
    plt.suptitle(title if title else exp_name, y=0.99)
    plt.tight_layout()
    
    fig.savefig(f"../plots/{exp_name}_no_labeling.png", dpi=300)
    texts = [[], [], []]
    for i, label in enumerate(all_means_dict['labels']):
            texts[0].append(axs[0].text(all_means_dict['compress_ratios'][i], all_means_dict['relative_errors'][i], label, fontsize=12))
            texts[1].append(axs[1].text(all_means_dict['compress_ratios'][i], all_means_dict['times'][i], label, fontsize=12))
            texts[2].append(axs[2].text(all_means_dict['compress_ratios'][i], all_means_dict['alphas'][i], label, fontsize=12))
    
    for i, t in enumerate(texts):
        adjust_text(t, expand=(1, 2), arrowprops=dict(arrowstyle="fancy", color='pink', lw=2), ax=axs[i]) 
    
    plt.show()
    fig.savefig(f"../plots/{exp_name}.png", dpi=300)
    
    return 


def plot_image_svd_reconstructions(U: np.ndarray,S: np.ndarray,Vt: np.ndarray, ranks: list, labels=None) -> None:
    fig, ax = plt.subplots(1,len(ranks),figsize=(16,(16 / Vt.shape[1] * U.shape[0] / len(ranks)).__ceil__()))
    
    compress_ratios = [calculate_2d_RSVD_compress_ratio(r, (U.shape[0], Vt.shape[1])) for r in ranks]
    for j, r in enumerate(ranks):
        # Construct approximate image
        Xapprox = U[:,:r+1] @ S[0:r+1,:r+1] @ Vt[:r+1,:]
        
        img = ax[j].imshow(Xapprox)
        img.set_cmap('gray')
        ax[j].axis('off')
        if labels is None:
            ax[j].set_title(f"r = {r}, c_r = {compress_ratios[j]:.3f}")
        else:
            ax[j].set_title(f"{labels[j]}, r = {r}, c_r = {compress_ratios[j]:.3f}")
        ax[j].title.set_fontsize(36 / len(ranks))
    plt.show()


def svd_rsvd_error_time_analysis(exp_name: str, X: np.ndarray, all_rsvd_dict: dict, truncated_svd_dict: dict = None, title: str = None) -> None:
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 18))
    qs = list(all_rsvd_dict.keys())
    ps = list(all_rsvd_dict[qs[0]].keys())
    
    linestyles = ["-", "--", "-."]
    markers = ["X", "s", "p", "*", "^", "P", "H", "D"]
    
    
    for i, q in enumerate(qs):
        linestyle = linestyles[i % 3]
        for j, p in enumerate(ps):
            marker = markers[j % 8]
            relative_errors = []
            times = []
            exact_compress_ratios = []
            compress_ratios = list(all_rsvd_dict[q][p].keys())
            
            for c_r in compress_ratios:
                relative_errors.append(all_rsvd_dict[q][p][c_r]["relative_error"])
                times.append(all_rsvd_dict[q][p][c_r]["method_time"])
                exact_compress_ratios.append(all_rsvd_dict[q][p][c_r]["exact_compress_ratio"])
                
            relative_errors = np.array(relative_errors)
            times = np.array(times)
            efficiency = compute_alpha(err=relative_errors, t=times)
            
            axs[0].plot(exact_compress_ratios, relative_errors, marker=marker, linestyle=linestyle, label=f"RSVD:{' q = ' + str(q)+',' if qs != [0] else ''} p={p}")
            axs[1].plot(exact_compress_ratios, times, marker=marker, linestyle=linestyle, label=f"RSVD:{' q = ' + str(q)+',' if qs != [0] else ''} p={p}")
            axs[2].plot(exact_compress_ratios, efficiency, marker=marker, linestyle=linestyle, label=f"RSVD:{' q = ' + str(q)+',' if qs != [0] else ''} p={p}")
    if truncated_svd_dict:
        relative_errors = []
        times = []
        exact_compress_ratios = []
        compress_ratios = list(truncated_svd_dict.keys())
        
        for c_r in compress_ratios:
            relative_errors.append(truncated_svd_dict[c_r]["relative_error"])
            times.append(truncated_svd_dict[c_r]["method_time"])
            exact_compress_ratios.append(truncated_svd_dict[c_r]["exact_compress_ratio"])
        
        relative_errors = np.array(relative_errors)
        times = np.array(times)
        efficiency = compute_alpha(err=relative_errors, t=times)
        
        axs[0].plot(exact_compress_ratios, relative_errors, marker='o', label=f"Truncated SVD", color="black")
        axs[1].plot(exact_compress_ratios, times, marker='o', label=f"Truncated SVD", color="black")
        axs[2].plot(exact_compress_ratios, efficiency, marker='o', label=f"Truncated SVD", color="black")
                
    axs[-1].set_xlabel("Compression Ratio")
    axs[0].set_title("Relative error of the reconstruction")
    axs[0].set_xscale("log") # for better visual interpretability: log scale?
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
    
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    
    axs[2].set_title("Efficiency")
    axs[2].set_ylabel("alpha = ln(1/error) / ln(e+time)")
    
    all_means_dict = prepare_mean_reconstruction_measurements_for_plot(X)
    
    for i, label in enumerate(all_means_dict['labels']):
        axs[0].scatter(all_means_dict['compress_ratios'][i], all_means_dict['relative_errors'][i], marker='D', color='pink', s=100)
        axs[1].scatter(all_means_dict['compress_ratios'][i], all_means_dict['times'][i], marker='D', color='pink', s=100)
        axs[2].scatter(all_means_dict['compress_ratios'][i], all_means_dict['alphas'][i], marker='D', color='pink', s=100)
      
    max_mean_error = np.max(all_means_dict['relative_errors'])    
    axs[0].set_ylim(-0.05 * max_mean_error, 1.05 * max_mean_error)
    
     # add manual symbols to auto legend
    handles, labels = axs[0].get_legend_handles_labels()
    point = Line2D([0], [0], label='M-(axes) - mean along axes', marker='D', color='pink', markersize=10, linestyle='')
    handles.extend([point])
    axs[0].legend(handles=handles, loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.3))
    
    plt.suptitle(title if title else exp_name, y=0.99)
    plt.tight_layout()
    
    fig.savefig(f"../plots/{exp_name}_no_labeling.png", dpi=300)
    texts = [[], [], []]
    for i, label in enumerate(all_means_dict['labels']):
            texts[0].append(axs[0].text(all_means_dict['compress_ratios'][i], all_means_dict['relative_errors'][i], label, fontsize=12))
            texts[1].append(axs[1].text(all_means_dict['compress_ratios'][i], all_means_dict['times'][i], label, fontsize=12))
            texts[2].append(axs[2].text(all_means_dict['compress_ratios'][i], all_means_dict['alphas'][i], label, fontsize=12))
    
    for i, t in enumerate(texts):
        adjust_text(t, expand=(1, 2), arrowprops=dict(arrowstyle="fancy", color='pink', lw=2), ax=axs[i]) 

    plt.show()
    fig.savefig(f"../plots/{exp_name}.png", dpi=300)
    return


def plot_GD_vs_energy_saving_cutoff_selection(title: str, sigmas: np.ndarray, tau: float, GD_rank: int, energy_frac: float) -> None:
    if energy_frac < 0 or energy_frac > 1:
        raise ValueError("energy_frac must be between 0 and 1")
    cumulated_energy = np.cumsum(sigmas) / np.sum(sigmas)
    rank_energy = np.argmax(cumulated_energy > energy_frac)
    
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].semilogy(sigmas, '-o')
    ax[0].axhline(tau, color='black', linestyle='--', label=f"tau (GD noise floor)")
    ax[0].axvline(GD_rank, color='r', label=f"GD_rank = {GD_rank}")
    ax[0].axvline(rank_energy, color='y', label=f"rank_90 = {rank_energy}")
    ax[0].set_title('Singular Values')
    ax[0].legend()
    
    ax[1].plot(cumulated_energy, '-o')
    ax[1].axvline(GD_rank, color='r', label=f"Gavish-Donoho cutoff --> {cumulated_energy[GD_rank]: .3f}")
    ax[1].axvline(rank_energy, color='y', label=f"rank_energy --> {energy_frac: .3f}")
    ax[1].set_title('Singular Values: Cumulative Sum')
    ax[1].legend()
    plt.suptitle(title)
    plt.show()
    return rank_energy