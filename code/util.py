import os, shutil
import pyttb.tensor as ttb

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