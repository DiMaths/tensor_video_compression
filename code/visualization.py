import matplotlib.pyplot as plt
import numpy as np

def error_time_analysis(exp_name: str, results_rsvd: dict = None, results_tucker: dict = None, results_hosvd: dict = None, show_plot: bool = False):
    labels = []
    colors = []
    if results_rsvd:
        comp_ratios = [int(key) for key in list(results_rsvd.keys())]
        for i in range(3):
            labels.append(f"RSVD along {i}-th dim")
        colors = ["black", "yellow", "green"]
        
        relative_errors = [results_rsvd[compress_ratio]["relative_errors"] for compress_ratio in results_rsvd.keys()]
        times = [results_rsvd[key]["times"] for key in results_rsvd.keys()]
        
        if results_tucker:
            labels.append(f"Tucker")
            colors.append("blue")
            for j, compress_ratio in enumerate(results_tucker.keys()):
                relative_errors[j].append(results_tucker[compress_ratio]["relative_error"])
                times[j].append(results_tucker[compress_ratio]["method_time"])
            
        if results_hosvd:
            labels.append(f"HOSVD")
            colors.append("gray")
            for j, compress_ratio in enumerate(results_hosvd.keys()):
                relative_errors[j].append(results_hosvd[compress_ratio]["relative_error"])
                times[j].append(results_hosvd[compress_ratio]["method_time"])
            
    elif results_tucker:
        comp_ratios = [int(key) for key in list(results_tucker.keys())]
        labels.append(f"Tucker")
        colors.append("blue")
        relative_errors = [[results_tucker[compress_ratio]["relative_error"]] for compress_ratio in results_tucker.keys()]
        times = [[results_tucker[compress_ratio]["method_time"]] for compress_ratio in results_tucker.keys()]
        if results_hosvd:
            labels.append(f"HOSVD")
            colors.append("gray")
            for j, compress_ratio in enumerate(results_hosvd.keys()):
                relative_errors[j].append(results_hosvd[compress_ratio]["relative_error"])
                times[j].append(results_hosvd[compress_ratio]["method_time"])
            
    else:
        raise ValueError("At least one of two (results_rsvd, results_tucker) must be given!")

    relative_errors = np.array(relative_errors)
    times = np.array(times)
    efficiency = np.array([1 / (relative_errors[:, i] * times[:, i]) for i in range(times.shape[1])]) 
    # for better visual interpretability: use / 10 ** rel_errors?

    # Plotting
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs[-1].set_xlabel("Compression Ratio")
    
    # axs[0].set_xscale("log") # for better visual interpretability: log scale?
    
    
    for i in range(times.shape[1]):
        axs[0].plot(comp_ratios, relative_errors[:, i], marker='o', label=labels[i], color=colors[i])
        axs[1].plot(comp_ratios, times[:, i], marker='o', label=labels[i], color=colors[i])
        axs[2].plot(comp_ratios, efficiency[i], marker='o', label=labels[i], color=colors[i])
    
    axs[0].set_title("Relative error of the reconstruction")
    
    axs[0].set_ylim(0, 1.05 * max(1, np.max(relative_errors)))
    axs[0].axhline(y=1, color='red', linestyle='--')
    axs[0].set_ylabel("||X - X_approx||_F / ||X||_F")
        
    axs[1].set_title("Time of reconstruction")
    axs[1].set_ylabel("seconds")
    #axs[1].set_ylim(0, 1.25 * max(1, np.max(times)))
    
    axs[2].set_title("Efficiency measured as time to relative_error ratio")
    axs[2].set_yscale("log")
    
    axs[0].legend(loc='upper center', ncols=times.shape[1], bbox_to_anchor=(0.5, 1.35))
    plt.tight_layout()
    fig.savefig(f"../plots/{exp_name}.png", dpi=300)
    if show_plot:
        plt.show()
    return relative_errors, times, efficiency


"""
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