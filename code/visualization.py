import matplotlib.pyplot as plt
import numpy as np

def error_time_analysis(exp_name: str, all_results: dict):
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
    
    return relative_errors, times, efficiency


"""
# OLD Implementation
def error_time_rsvd_analysis(exp_name: str, results_rsvd: dict, show_plot: bool = False):
    
    relative_errors = np.array([results_rsvd[compress_ratio]["relative_errors"] for compress_ratio in results_rsvd.keys()])
    times = np.array([results_rsvd[key]["times"] for key in results_rsvd.keys()])
    efficiency = np.array([1 / (relative_errors[:, i] * times[:, i]) for i in range(3)]) # for better visual interpretability: use / 10 ** rel_errors?
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs[-1].set_xlabel("Compression Ratio")
    
    # axs[0].set_xscale("log") # for better visual interpretability: log scale?
    
    # comp_ratios = np.array([results_rsvd[compress_ratio]["exact_comp_ratios"] for compress_ratio in results_rsvd.keys()])
    # and comp_ratios[:, i] later for super accuracy to avoid discritization error of compression ratio
    # usually very small diff between desired and real compression ratio
    comp_ratios = [int(key) for key in list(results_rsvd.keys())]
    colors = ["black", "gray", "green"]
    for i in range(3):
        axs[0].plot(comp_ratios, relative_errors[:, i], marker='o', label=f"RSVD along {i}-th dim", color=colors[i])
        axs[1].plot(comp_ratios, times[:, i], marker='o', label=f"RSVD along {i}-th dim", color=colors[i])
        axs[2].plot(comp_ratios, efficiency[i], marker='o', label=f"RSVD along {i}-th dim", color=colors[i])
    
    axs[0].set_title("Relative error of the reconstruction")
    
    axs[0].set_ylim(0, 1.25 * max(1, np.max(relative_errors)))
    axs[0].axhline(y=1, color='red', linestyle='--')
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
        
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    
    axs[2].set_title("Efficiency measured as time to relative_error ratio")
    
    axs[0].legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.02))
    axs[1].legend(loc='upper center', ncols=3)
    axs[2].legend(loc='upper center', ncols=3)
    plt.tight_layout()
    fig.savefig(f"../plots/RSVD_{exp_name}.png", dpi=300)
    if show_plot:
        plt.show()
    return relative_errors, times, efficiency
"""