"""DeltaTracker wrapper for 3D dense tracking from video.

This module provides a wrapper class for the DeltaTracker model, making it easy to
perform 3D dense tracking on videos with optional depth estimation using UniDepth or DepthCrafter.

The data_root_path should be a directory containing one of the following:
    1. A single .mp4 video file
    2. A 'color' subdirectory containing image sequences (.png, .jpg, or .jpeg)
    3. Optionally, a 'depth' subdirectory with corresponding depth maps
    4. Optionally, precomputed depth files:
       - depth_pred.npy: UniDepth predictions
       - depth_depthcrafter.npy: DepthCrafter predictions

Example usage as a command-line tool:
    ```bash
    python delta_wrapper.py \\
        --ckpt checkpoints/densetrack3d.pth \\
        --data_root_path demo_data/rollerblade \\
        --output_path results/demo
    ```

Example usage as a module:
    ```python
    from delta_wrapper import DeltaTrackerWrapper
    
    tracker = DeltaTrackerWrapper(ckpt_path="checkpoints/densetrack3d.pth")
    results = tracker.process_video(
        data_root_path="demo_data/rollerblade",
        output_path="results/demo"
    )
    ```

For inference requiring less memory, use:
- `--upsample_factor 8`: Increases the stride between frames, trading off accuracy for speed
- `--use_fp16`: Enables half-precision (FP16) computation for faster inference with slightly reduced precision
"""

import os
import pickle
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict, Union, Tuple
import argparse

from densetrack3d.datasets.custom_data import read_data_with_depthcrafter
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.utils.depthcrafter_utils import read_video
from densetrack3d.utils.visualizer import Visualizer


class DeltaTrackerWrapper:
    def __init__(
        self,
        ckpt_path: str,
        use_fp16: bool = False,
        model_resolution: Tuple[int, int] = (384, 512),
        window_len: int = 16,
        stride: int = 4,
        upsample_factor: int = 4,
        device: str = "cuda",
        gpu_id: int = 0,
        debug: bool = False
    ):
        """Initialize DeltaTracker with model parameters and settings.

        Args:
            ckpt_path: Path to the DenseTrack3D checkpoint
            use_fp16: Whether to use FP16 precision
            model_resolution: Resolution for the model (height, width)
            window_len: Window length for tracking
            stride: Model stride
            upsample_factor: Upsampling factor
            device: Device to run the model on ('cuda' or 'cpu')
            gpu_id: GPU device ID to use when device is 'cuda' (default: 0)
            debug: Whether to print debug information (default: False)
        """
        self.debug = debug
        if self.debug:
            print("Initializing DeltaTrackerWrapper...")
        
        if device == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                if gpu_id >= torch.cuda.device_count():
                    print(f"GPU {gpu_id} not available, using GPU 0")
                    gpu_id = 0
                torch.cuda.set_device(gpu_id)
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name()}")
        
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.base_dir = os.getcwd()
        
        # Initialize DenseTrack3D model
        if self.debug:
            print("Loading DenseTrack3D model...")
        self.model = DenseTrack3D(
            stride=stride,
            window_len=window_len,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=model_resolution,
            upsample_factor=upsample_factor
        )
        
        # Load checkpoint
        if self.debug:
            print(f"Loading checkpoint from {ckpt_path}...")
        with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)
        
        # Initialize predictor
        if self.debug:
            print("Initializing predictor...")
        self.predictor = DensePredictor3D(model=self.model)
        self.predictor = self.predictor.eval().to(self.device)
        
        if self.debug:
            print("Initialization complete!")

    @torch.inference_mode()
    def predict_unidepth(self, video: np.ndarray) -> np.ndarray:
        """Predict depth using UniDepth model.

        Args:
            video: Input video array of shape (T, H, W, C)

        Returns:
            Predicted depth maps of shape (T, H, W)
        """
        if self.debug:
            print("Starting UniDepth prediction...")
        
        # Import UniDepth here to avoid dependency if not used
        os.sys.path.append(os.path.abspath(os.path.join(self.base_dir, "submodules", "UniDepth")))
        from unidepth.models import UniDepthV2
        
        if self.debug:
            print("Loading UniDepth model...")
        unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        unidepth_model = unidepth_model.eval().to(self.device)
        
        video_torch = torch.from_numpy(video).permute(0, 3, 1, 2).to(self.device)
        depth_pred = []
        
        if self.debug:
            print(f"Processing video chunks ({len(video)} frames)...")
        chunks = torch.split(video_torch, 32, dim=0)
        for i, chunk in enumerate(chunks):
            if self.debug:
                print(f"Processing chunk {i+1}/{len(chunks)}...")
            predictions = unidepth_model.infer(chunk)
            depth_pred_ = predictions["depth"].squeeze(1).cpu().numpy()
            depth_pred.append(depth_pred_)
            
        if self.debug:
            print("UniDepth prediction complete!")
        return np.concatenate(depth_pred, axis=0)

    @torch.inference_mode()
    def predict_depthcrafter(self, video: np.ndarray) -> np.ndarray:
        """Predict depth using DepthCrafter model.

        Args:
            video: Input video array of shape (T, H, W, C)

        Returns:
            Predicted depth maps of shape (T, H, W)
        """
        if self.debug:
            print("Starting DepthCrafter prediction...")
        
        os.sys.path.append(os.path.abspath(os.path.join(self.base_dir, "submodules", "DepthCrafter")))
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

        if self.debug:
            print("Loading DepthCrafter model...")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            "tencent/DepthCrafter",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        pipe = DepthCrafterPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not enabled: {e}")
        pipe.enable_attention_slicing()

        if self.debug:
            print("Processing video frames...")
        frames, ori_h, ori_w = read_video(video, max_res=1024)
        res = pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=1.2,
            num_inference_steps=25,
            window_size=110,
            overlap=25,
            track_time=False,
        ).frames[0]

        if self.debug:
            print("Post-processing depth maps...")
        res = res.sum(-1) / res.shape[-1]
        res = (res - res.min()) / (res.max() - res.min())
        res = F.interpolate(
            torch.from_numpy(res[:, None]), 
            (ori_h, ori_w), 
            mode="nearest"
        ).squeeze(1).numpy()
        
        if self.debug:
            print("DepthCrafter prediction complete!")
        return res

    def process_video(
        self,
        data_root_path: str,
        output_path: str,
        use_depthcrafter: bool = False,
        viz_sparse: bool = True,
        downsample: int = 16,
        use_gt_depth: bool = False
    ) -> Dict:
        """Process a video file and generate tracking results.

        Args:
            data_root_path: Root directory containing either an MP4 file or image sequence
            output_path: Path to save results
            use_depthcrafter: Whether to use DepthCrafter for depth estimation
            viz_sparse: Whether to visualize sparse tracking
            downsample: Downsample factor for sparse tracking visualization
            use_gt_depth: Whether to use ground truth depth from depths.npy or depth folder

        Returns:
            Dictionary containing tracking results
        """
        # Read video data
        video, videodepth, videodisp = read_data_with_depthcrafter(
            full_path=data_root_path,
            use_gt_depth=use_gt_depth
        )
        
        # Get depth if not provided and not using ground truth
        if videodepth is None and not use_gt_depth:
            print("Running UniDepth for depth estimation...")
            videodepth = self.predict_unidepth(video)
            np.save(os.path.join(data_root_path, "depth_pred.npy"), videodepth)
        
        # Use DepthCrafter if specified
        if use_depthcrafter:
            if videodisp is None:
                print("Running DepthCrafter...")
                videodisp = self.predict_depthcrafter(video)
                np.save(os.path.join(data_root_path, "depth_depthcrafter.npy"), videodisp)
            videodepth = least_square_align(videodepth, videodisp)

        # Prepare tensors
        video = torch.from_numpy(video).permute(0, 3, 1, 2).to(self.device)[None].float()
        videodepth = torch.from_numpy(videodepth).unsqueeze(1).to(self.device)[None].float()

        # Create output directory
        vid_name = os.path.basename(data_root_path)
        save_dir = os.path.join(output_path, vid_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving results to {save_dir}")

        # Run DenseTrack3D
        print("Running DenseTrack3D...")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_fp16):
            out_dict = self.predictor(
                video,
                videodepth,
                grid_query_frame=0,
            )

        # Process results
        trajs_3d_dict = {k: v[0].cpu().numpy() for k, v in out_dict["trajs_3d_dict"].items()}
        
        # Save tracking results
        with open(os.path.join(save_dir, "dense_3d_track.pkl"), "wb") as handle:
            pickle.dump(trajs_3d_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Visualize if requested
        if viz_sparse:
            self._visualize_sparse_tracking(
                video, out_dict, save_dir, downsample
            )

        return trajs_3d_dict

    def _visualize_sparse_tracking(
        self,
        video: torch.Tensor,
        out_dict: Dict,
        save_dir: str,
        downsample: int
    ) -> None:
        """Visualize sparse tracking results.

        Args:
            video: Input video tensor
            out_dict: Dictionary containing tracking results
            save_dir: Directory to save visualization
            downsample: Downsample factor for visualization
        """
        print("Visualizing sparse 2D tracking...")
        W = video.shape[-1]
        visualizer_2d = Visualizer(
            save_dir="results/demo",
            fps=10,
            show_first_frame=0,
            linewidth=int(1 * W / 512),
            tracks_leave_trace=10
        )

        trajs_uv = out_dict["trajs_uv"]
        trajs_vis = out_dict["vis"]
        dense_reso = out_dict["dense_reso"]

        # Process trajectories for visualization
        sparse_trajs_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_uv = sparse_trajs_uv[:, :, ::downsample, ::downsample]
        sparse_trajs_uv = rearrange(sparse_trajs_uv, "b t h w c -> b t (h w) c")

        sparse_trajs_vis = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_vis = sparse_trajs_vis[:, :, ::downsample, ::downsample]
        sparse_trajs_vis = rearrange(sparse_trajs_vis, "b t h w -> b t (h w)")

        # Generate and save visualization
        video2d_viz = visualizer_2d.visualize(
            video,
            sparse_trajs_uv,
            sparse_trajs_vis[..., None],
            filename="demo",
            save_video=False
        )

        video2d_viz = video2d_viz[0].permute(0, 2, 3, 1).cpu().numpy()
        media.write_video(os.path.join(save_dir, "sparse_2d_track.mp4"), video2d_viz, fps=10) 

def get_parser() -> argparse.ArgumentParser:
    """Create and return the command line argument parser.

    Returns:
        ArgumentParser object with all the command line arguments
    """
    parser = argparse.ArgumentParser(description="DeltaTracker: 3D Dense Tracking from Video")
    
    # Required arguments
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the DenseTrack3D checkpoint")
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True, 
        help="Root directory containing input data. Must contain either: "
             "(1) a single .mp4 file, or "
             "(2) a 'color' subdirectory with image sequences. "
             "May optionally contain a 'depth' directory with depth maps "
             "or cached depth predictions (depth_pred.npy, depth_depthcrafter.npy)"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
    
    # Optional arguments
    parser.add_argument("--use_depthcrafter", action="store_true", help="Use DepthCrafter for depth estimation")
    parser.add_argument("--no_viz_sparse", action="store_true", help="Disable sparse tracking visualization")
    parser.add_argument("--downsample", type=int, default=16, help="Downsample factor for sparse tracking visualization")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision for inference")
    parser.add_argument("--model_height", type=int, default=384, help="Model input height resolution")
    parser.add_argument("--model_width", type=int, default=512, help="Model input width resolution")
    parser.add_argument("--window_len", type=int, default=16, help="Window length for tracking")
    parser.add_argument("--stride", type=int, default=4, help="Model stride")
    parser.add_argument("--upsample_factor", type=int, default=4, help="Upsampling factor")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use when device is 'cuda'")
    parser.add_argument("--debug", action="store_true", help="Print debug information during execution")
    parser.add_argument("--use_gt_depth", action="store_true", help="Use ground truth depth from depths.npy or depth folder")
    
    return parser


def main():
    """Main function to run the DeltaTracker from command line."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = DeltaTrackerWrapper(
        ckpt_path=args.ckpt,
        use_fp16=args.use_fp16,
        model_resolution=(args.model_height, args.model_width),
        window_len=args.window_len,
        stride=args.stride,
        upsample_factor=args.upsample_factor,
        device=args.device,
        gpu_id=args.gpu_id,
        debug=args.debug
    )
    
    # Process video
    results = tracker.process_video(
        data_root_path=args.data_root_path,
        output_path=args.output_path,
        use_depthcrafter=args.use_depthcrafter,
        viz_sparse=not args.no_viz_sparse,
        downsample=args.downsample,
        use_gt_depth=args.use_gt_depth
    )
    
    print("Processing completed successfully!")
    print(results)
    return results


if __name__ == "__main__":
    main()