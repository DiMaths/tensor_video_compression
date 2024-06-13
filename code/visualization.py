import matplotlib.pyplot as plt
import numpy as np

from rsvd import calculate_2d_RSVD_compression_ratio
    
def error_time_analysis(exp_name: str, all_results: dict) -> None:
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
            exact_comp_ratios = []
            for j, compress_ratio in enumerate(all_results[method].keys()):
                relative_errors.append(all_results[method][compress_ratio]["relative_error"])
                exact_comp_ratios.append(all_results[method][compress_ratio]["exact_compression_ratio"])
                times.append(all_results[method][compress_ratio]["method_time"])
                
        
            relative_errors = np.array(relative_errors)
            exact_comp_ratios = np.array(exact_comp_ratios)
            times = np.array(times)
            efficiency = np.array(1 / (relative_errors * times))
            
            axs[0].plot(exact_comp_ratios, relative_errors, marker='o', label=method, color=color)
            axs[1].plot(exact_comp_ratios, times, marker='o', label=method, color=color)
            axs[2].plot(exact_comp_ratios, efficiency, marker='o', label=method, color=color)
            
        else:
            num_labels += 3
            results_rsvd = all_results[method]
            relative_errors = np.array([results_rsvd[compress_ratio]["relative_error"] for compress_ratio in results_rsvd.keys()])
            times = np.array([results_rsvd[key]["method_time"] for key in results_rsvd.keys()])
            efficiency = np.array([1 / (relative_errors[:, i] * times[:, i]) for i in range(3)])
            
            exact_comp_ratios = np.array([results_rsvd[compress_ratio]["exact_compression_ratio"] for compress_ratio in results_rsvd.keys()])
            colors = ["black", "yellow", "green"]
            for i in range(3):
                axs[0].plot(exact_comp_ratios[:, i], relative_errors[:, i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                axs[1].plot(exact_comp_ratios[:, i], times[:, i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                axs[2].plot(exact_comp_ratios[:, i], efficiency[i], marker='o', label=f"RSVD {i}-th dim", color=colors[i])
                
    axs[-1].set_xlabel("Compression Ratio")
    axs[0].set_title("Relative error of the reconstruction")
    axs[0].set_xscale("log") # for better visual interpretability: log scale?
    axs[0].set_ylim(0, 1.05 * max(1, np.max(axs[0].get_ylim())))
    
    axs[0].axhline(y=1, color='red', linestyle='-')
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
    
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    
    axs[2].set_title("Efficiency")
    axs[2].set_ylabel("1 / (time * relative_error)")
    # axs[2].set_yscale("log")
    
    axs[0].legend(loc='upper center', ncols=num_labels, bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()
    plt.show()
    
    fig.savefig(f"../plots/{exp_name}.png", dpi=300)
    return 


def plot_image_svd_reconstructions(U: np.ndarray,S: np.ndarray,Vt: np.ndarray, ranks: list, labels=None) -> None:
    fig, ax = plt.subplots(1,len(ranks),figsize=(16,(16 / Vt.shape[1] * U.shape[0] / len(ranks)).__ceil__()))
    
    for j, r in enumerate(ranks):
        # Construct approximate image
        Xapprox = U[:,:r+1] @ S[0:r+1,:r+1] @ Vt[:r+1,:]
        
        img = ax[j].imshow(Xapprox)
        img.set_cmap('gray')
        ax[j].axis('off')
        if labels is None:
            ax[j].set_title('r = ' + str(r))
        else:
            ax[j].set_title(labels[j] + ' (r = ' + str(r) + ')')
            
    plt.show()


def svd_rsvd_error_time_analysis(exp_name: str, all_rsvd_dict: dict, truncated_svd_dict: dict = None) -> None:
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 18))
    qs = list(all_rsvd_dict.keys())
    ps = list(all_rsvd_dict[qs[0]].keys())
    ranks = list(all_rsvd_dict[qs[0]][ps[0]].keys())
    linestyles = ["-", "--", "-."]
    markers = ["X", "s", "p", "*", "^", "P", "H", "D"]
    exact_comp_ratios = [calculate_2d_RSVD_compression_ratio(r, all_rsvd_dict[qs[0]][ps[0]][ranks[0]]['approximation'].shape) for r in ranks]
    
    for i, q in enumerate(qs):
        linestyle = linestyles[i % 3]
        for j, p in enumerate(ps):
            marker = markers[j % 8]
            relative_errors = []
            times = []
            for r in ranks:
                relative_errors.append(all_rsvd_dict[q][p][r]["relative_error"])
                times.append(all_rsvd_dict[q][p][r]["method_time"])
                
            relative_errors = np.array(relative_errors)
            times = np.array(times)
            efficiency = np.array(1 / (relative_errors * times))
            
            axs[0].plot(exact_comp_ratios, relative_errors, marker=marker, linestyle=linestyle, label=f"RSVD: q={q}, p={p}")
            axs[1].plot(exact_comp_ratios, times, marker=marker, linestyle=linestyle, label=f"RSVD: q={q}, p={p}")
            axs[2].plot(exact_comp_ratios, efficiency, marker=marker, linestyle=linestyle, label=f"RSVD: q={q}, p={p}")
    if truncated_svd_dict:
        relative_errors = []
        times = []
        for r in ranks:
            relative_errors.append(truncated_svd_dict[r]["relative_error"])
            times.append(truncated_svd_dict[r]["method_time"])
                
        
        relative_errors = np.array(relative_errors)
        times = np.array(times)
        efficiency = np.array(1 / (relative_errors * times))
        
        axs[0].plot(exact_comp_ratios, relative_errors, marker='o', label=f"Truncated SVD", color="black")
        axs[1].plot(exact_comp_ratios, times, marker='o', label=f"Truncated SVD", color="black")
        axs[2].plot(exact_comp_ratios, efficiency, marker='o', label=f"Truncated SVD", color="black")
                
    axs[-1].set_xlabel("Compression Ratio")
    axs[0].set_title("Relative error of the reconstruction")
    axs[0].set_xscale("log") # for better visual interpretability: log scale?
    axs[0].set_ylim(0, 1.05 * max(1, np.max(axs[0].get_ylim())))
    
    axs[0].axhline(y=1, color='red', linestyle='-')
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
    
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    
    axs[2].set_title("Efficiency")
    axs[2].set_ylabel("1 / (time * relative_error)")
    # axs[2].set_yscale("log")
    
    axs[0].legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
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