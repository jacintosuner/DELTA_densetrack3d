import glob
import io
import json
import os
import pickle
from typing import Mapping, Tuple, Union

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int64, Shaped
from PIL import Image
from torch import Tensor, nn
from pyk4a import PyK4APlayback
from pyk4a.calibration import CalibrationType


def read_mkv_data(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read video and depth data from an MKV file recorded with Azure Kinect.
    
    Args:
        video_path: Path to the MKV file
        
    Returns:
        Tuple containing:
            - video: numpy array of shape (T, H, W, 3) containing RGB frames
            - videodepth: numpy array of shape (T, H, W) containing depth frames
    """
    playback = PyK4APlayback(video_path)
    playback.open()
    
    video_frames = []
    depth_frames = []
    
    try:
        while True:
            capture = playback.get_next_capture()
            
            # Get RGB and depth frames
            rgb_frame = capture.color[:, :, :3][:, :, ::-1]  # BGR to RGB
            depth_frame = capture.transformed_depth
            
            if rgb_frame is not None and depth_frame is not None:
                video_frames.append(rgb_frame)
                depth_frames.append(depth_frame)
                
    except EOFError:
        pass  # End of video file reached
        
    playback.close()
    
    if len(video_frames) > 0:
        video = np.stack(video_frames)
        videodepth = np.stack(depth_frames)
        return video, videodepth
    else:
        return None, None


def read_data(data_root="demo_data", name="rollerblade", full_path=None):
    if full_path is None:
        full_path = os.path.join(data_root, name)
    
    # First try to read MKV file if present
    mkv_path = glob.glob(f"{full_path}/*.mkv")
    if len(mkv_path) > 0:
        print(f"Read MKV video: {mkv_path[0]}")
        return read_mkv_data(mkv_path[0])
    
    # Otherwise try MP4 or image sequence
    video_path = glob.glob(f"{full_path}/*.mp4")
    if len(video_path) > 0:
        print(f"Read video: {video_path[0]}")
        video = media.read_video(video_path[0])
    elif os.path.isdir(os.path.join(full_path, "color")):
        rgb_folder = os.path.join(full_path, "color")
        rgb_frames = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        video = []

        for rgb_frame in rgb_frames:
            img = cv2.imread(os.path.join(rgb_folder, rgb_frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)

        video = np.stack(video)

    if os.path.exists(os.path.join(full_path, "depth")):
        videodepth = []
        depth_names = sorted(os.listdir(os.path.join(full_path, "depth")))
        for depth_name in depth_names:
            depth = cv2.imread(os.path.join(full_path, "color", depth_name), cv2.IMREAD_ANYDEPTH)
            videodepth.append(depth)

        if len(videodepth) > 0 and len(video) == len(videodepth):
            videodepth = np.stack(videodepth)
        else:
            videodepth = None

    elif os.path.exists(os.path.join(full_path, "depth_pred.npy")):
        videodepth = np.load(os.path.join(full_path, "depth_pred.npy"))
    else:
        videodepth = None

    return video, videodepth


def read_data_with_depthcrafter(data_root="demo_data", name="rollerblade", full_path=None):
    if full_path is None:
        full_path = os.path.join(data_root, name)

    video, videodepth = read_data(data_root, name, full_path)

    if os.path.exists(os.path.join(full_path, "depth_depthcrafter.npy")):
        videodisp = np.load(os.path.join(full_path, "depth_depthcrafter.npy"))
    else:
        videodisp = None

    return video, videodepth, videodisp


def resize_video(
    video: Float[Tensor, "b t c h w"],
    videodepth: Float[Tensor, "b t 1 h w"] = None,
    # target_w: int = 512,
) -> Float[Tensor, "b t c h w"] | tuple[Float[Tensor, "b t c h w"], Float[Tensor, "b t 1 h w"]]:

    B, T, *_, ori_h, ori_w = video.shape

    ref_ratio = float(384 / 512)
    ori_ratio = float(ori_h / ori_w)
    if ori_ratio < ref_ratio:
        target_w = 512
        target_h = int(target_w * ori_h / ori_w)
    elif ori_ratio > ref_ratio:
        target_h = 384
        target_w = int(target_h * ori_w / ori_h)
    else:
        target_h, target_w = 384, 512

    video_resized = F.interpolate(
        rearrange(video, "b t c h w -> (b t) c h w"), size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=B, t=T)

    if videodepth is not None:
        if videodepth.shape[-2] == 384 and videodepth.shape[-1] == 512:
            videodepth_resized = videodepth
        else:
            videodepth_resized = F.interpolate(
                rearrange(videodepth, "b t 1 h w -> (b t) 1 h w"),
                size=(target_h, target_w),
                mode="nearest",
            )
            videodepth_resized = rearrange(videodepth_resized, "(b t) 1 h w -> b t 1 h w", b=B, t=T)

        return video_resized, videodepth_resized

    return video_resized


def read_iphone_data(data_root, name):
    data_dir = os.path.join(data_root, name)
    depth_type = "depth_anything_colmap"
    factor = 1
    start = 0
    end = -1
    split = "train"
    with open(os.path.join(data_dir, "splits", f"{split}.json")) as f:
        split_dict = json.load(f)
    full_len = len(split_dict["frame_names"])
    end = min(end, full_len) if end > 0 else full_len
    frame_names = split_dict["frame_names"][start:end]

    video = np.stack(
        [iio.imread(os.path.join(data_dir, f"rgb/{factor}x/{frame_name}.png"))[..., :3] for frame_name in frame_names],
        axis=0,
    )

    videodepth = []
    for frame_name in frame_names:
        depth = np.load(
            os.path.join(
                data_dir,
                f"flow3d_preprocessed/aligned_{depth_type}/",
                f"{factor}x/{frame_name}.npy",
            )
        )
        depth[depth < 1e-3] = 1e-3
        depth = 1.0 / depth

        videodepth.append(depth)
    videodepth = np.stack(videodepth, axis=0)

    # NOTE convert portrait to landscape
    video = np.ascontiguousarray(np.transpose(video, (0, 2, 1, 3))[:, :, ::-1, :])
    videodepth = np.ascontiguousarray(np.transpose(videodepth, (0, 2, 1))[:, :, ::-1])

    return video, videodepth


def read_iphone_data_monst3r(data_root):

    # frame_paths = sorted(glob.glob(f"{data_root}/{name}/frame_*.png"))
    rgb_paths = sorted(glob.glob(f"{data_root}/frame_*.png"))
    depth_paths = sorted(glob.glob(f"{data_root}/frame_*.npy"))

    video, videodepth = [], []
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        frame = iio.imread(rgb_path)
        depth = np.load(depth_path)
        video.append(frame)
        videodepth.append(depth)
    video = np.stack(video)
    videodepth = np.stack(videodepth)

    # video = np.ascontiguousarray(np.transpose(video, (0, 2, 1, 3))[:, :, ::-1, :])
    # videodepth = np.ascontiguousarray(np.transpose(videodepth, (0, 2, 1))[:, :, ::-1])

    return video, videodepth
