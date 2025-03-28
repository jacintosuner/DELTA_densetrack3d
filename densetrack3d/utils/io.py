import argparse
import json
import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F


def read_pickle(file_path):
    try:
        # Open the pickle file in binary read mode
        with open(file_path, "rb") as file:
            # Load the pickle data
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None


def create_folder(path, verbose=False, exist_ok=True, safe=True):
    if os.path.exists(path) and not exist_ok:
        if not safe:
            raise OSError
        return False
    try:
        os.makedirs(path)
    except:
        if not safe:
            raise OSError
        return False
    if verbose:
        print(f"Created folder: {path}")
    return True


def read_video(path, start_step=0, time_steps=None, channels="first", exts=("jpg", "png"), resolution=None):
    if path.endswith(".mp4"):
        video = read_video_from_file(path, start_step, time_steps, channels, resolution)
    else:
        video = read_video_from_folder(path, start_step, time_steps, channels, resolution, exts)
    return video


def read_videodepth(path, start_step=0, time_steps=None, channels="first", exts=("jpg", "png"), resolution=None):
    # if path.endswith(".mp4"):
    #     video = read_video_from_file(path, start_step, time_steps, channels, resolution)
    # else:
    video = read_videodepth_from_folder(path, start_step, time_steps, channels, resolution, exts)
    return video


def read_video_from_file(path, start_step, time_steps, channels, resolution):
    video, _, _ = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")
    if time_steps is None:
        time_steps = len(video) - start_step
    video = video[start_step : start_step + time_steps]
    if resolution is not None:
        video = F.interpolate(video, size=resolution, mode="bilinear")
    if channels == "last":
        video = video.permute(0, 2, 3, 1)
    video = video / 255.0
    return video


def read_video_from_folder(path, start_step, time_steps, channels, resolution, exts):
    paths = []
    for ext in exts:
        paths += glob(os.path.join(path, f"*.{ext}"))
    paths = sorted(paths)
    if time_steps is None:
        time_steps = len(paths) - start_step
    video = []
    for step in range(start_step, start_step + time_steps):
        frame = read_frame(paths[step], resolution, channels)
        video.append(frame)
    video = torch.stack(video)
    return video


def read_frame(path, resolution=None, channels="first"):
    frame = Image.open(path).convert("RGB")
    frame = np.array(frame)
    frame = frame.astype(np.float32)
    frame = frame / 255
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)
    if resolution is not None:
        frame = F.interpolate(frame[None], size=resolution, mode="bilinear")[0]
    if channels == "last":
        frame = frame.permute(1, 2, 0)
    return frame


def read_videodepth_from_folder(path, start_step, time_steps, channels, resolution, exts):
    paths = []
    for ext in exts:
        paths += glob(os.path.join(path, f"*.{ext}"))
    paths = sorted(paths)
    if time_steps is None:
        time_steps = len(paths) - start_step
    videodepth = []
    for step in range(start_step, start_step + time_steps):
        depthframe = read_depth_frame(paths[step], resolution, channels)
        videodepth.append(depthframe)
    videodepth = torch.stack(videodepth)
    return videodepth


def read_depth_frame(path, resolution=None, channels="first"):
    depth16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depth = depth16.astype(np.float32)

    frame = torch.from_numpy(depth)[..., None]
    frame = frame.permute(2, 0, 1)
    if resolution is not None:
        frame = F.interpolate(frame[None], size=resolution, mode="nearest")[0]
    if channels == "last":
        frame = frame.permute(1, 2, 0)
    return frame


def write_video(video, path, channels="first", zero_padded=True, ext="png", dtype="torch"):
    if dtype == "numpy":
        video = torch.from_numpy(video)
    if path.endswith(".mp4"):
        write_video_to_file(video, path, channels)
    else:
        write_video_to_folder(video, path, channels, zero_padded, ext)


def write_video_to_file(video, path, channels):
    create_folder(os.path.dirname(path))
    if channels == "first":
        video = video.permute(0, 2, 3, 1)
    video = (video.cpu() * 255.0).to(torch.uint8)
    torchvision.io.write_video(path, video, 24, "h264", options={"pix_fmt": "yuv420p", "crf": "23"})
    return video


def write_video_to_folder(video, path, channels, zero_padded, ext):
    create_folder(path)
    time_steps = video.shape[0]
    for step in range(time_steps):
        pad = "0" * (len(str(time_steps)) - len(str(step))) if zero_padded else ""
        frame_path = os.path.join(path, f"{pad}{step}.{ext}")
        write_frame(video[step], frame_path, channels)


def write_frame(frame, path, channels="first"):
    create_folder(os.path.dirname(path))
    frame = frame.cpu().numpy()
    if channels == "first":
        frame = np.transpose(frame, (1, 2, 0))
    frame = np.clip(np.round(frame * 255), 0, 255).astype(np.uint8)
    frame = Image.fromarray(frame)
    frame.save(path)


def write_frame_np(frame, path, channels="first"):
    create_folder(os.path.dirname(path))
    # frame = frame.cpu().numpy()
    if channels == "first":
        frame = np.transpose(frame, (1, 2, 0))
    frame = np.clip(np.round(frame * 255), 0, 255).astype(np.uint8)
    frame = Image.fromarray(frame)
    frame.save(path)


def read_tracks(path, allow_pickle=False):
    return np.load(path, allow_pickle=allow_pickle)


def write_tracks(tracks, path):
    np.save(path, tracks)


def read_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    return args


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("predictor."):
                    continue

                new_state_dict[k.replace("model.", "")] = v
            state_dict = new_state_dict

    return state_dict
