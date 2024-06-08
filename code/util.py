import os, shutil
import pyttb.tensor as ttb
import json

def empty_the_dir(dir_path):
    with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)


def load_ttb_from_npy_file(path_to_npy):
    with open(path_to_npy, 'rb') as f:
        tensor = np.load(f)
    return ttb.from_data(tensor)


def decomposition_results_to_json(exp_name, method, results_dict):
    # dump only serializable parts (ignore tensors)
    json_res = dict()
    for key in results_dict.keys():
        json_res[key] = dict()
        for non_series_key in ['relative_error', 'method_time', 'exact_compression_ratio']:
            json_res[key][non_series_key] = results_dict[key][non_series_key]
    json_object = json.dumps(json_res, indent=4) 
    
    with open(f"../jsons/{method}_{exp_name}.json", "w") as outfile:
        outfile.write(json_object)
        print(f"Saved to ../jsons/{method}_{exp_name}.json")