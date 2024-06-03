import numpy as np
import os
import time
import cv2
from tqdm.auto import tqdm

def to_image_format(tensor):
    """A convenience function to carefully convert from a float dtype back to uint8"""
    if type(tensor) != np.ndarray:
        raise TypeError("tensor is not numpy ndarray")
        
    if tensor.dtype == np.uint8:
        return tensor
    im = tensor
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)
    
def read_video_as_numpy(path, n_frames, colorful=True, save_to=None):
    source = cv2.VideoCapture(path)
    tensor = []
    i = 0
    ret = True
    for i in tqdm(range(n_frames), desc=f"Reading first {n_frames} frames from {path}"):
        if not ret:
            print("Video has fewer frames than requested, read the full video.")
            break
        ret, frame = source.read()
        if colorful:
            tensor.append(frame)
        else:
            tensor.append(np.mean(frame, axis=2))
        i += 1
    source.release()
    tensor = np.array(tensor, dtype=np.float64)
    if save_to is not None:
        with open(save_to, 'wb') as f:
            np.save(f, tensor)
    print(f"Shape of read tensor = {tensor.shape}")
    return tensor

def save_numpy_as_video(tensor, path, fps, format_ = 'XVID', colorful=True):
    frame_size = tensor.shape[1:3][::-1]
    tensor = to_image_format(tensor)
    
    output = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*format_), fps, frame_size, colorful)
    if not output.isOpened():
        raise ValueError("Error: Could not create the output video file.")
    
    for i in tqdm(range(tensor.shape[0]), desc=f"Writing {frame_size}-frames\n into {path}"):
        output.write(tensor[i].astype(np.uint8))
    output.release()
    cv2.destroyAllWindows()

def convert_npy_to_avi(take_from, save_to, fps, format_ = 'XVID', colorful=True):
    with open(take_from, 'rb') as f:
        tensor = np.load(f)
    
    frame_size = tensor.shape[1:3][::-1]
    output = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*format_), fps, frame_size, colorful)
    if not output.isOpened():
        raise ValueError("Error: Could not create the output video file.")
    
    for i in tqdm(range(tensor.shape[0]), desc=f"Writing {frame_size}-frames\n into {save_to} from {take_from}"):
        output.write(tensor[i])
    output.release()
    cv2.destroyAllWindows()
    