"""
Script to extract depth data from MKV files recorded with Azure Kinect and save them as NumPy arrays.
"""

import numpy as np
from pathlib import Path
import argparse
from pyk4a import PyK4APlayback
from tqdm import tqdm

class MKVDepthExtractor:
    def __init__(self, mkv_path):
        """
        Initialize the MKV depth extractor.
        
        Args:
            mkv_path (str): Path to the input MKV file
        """
        self.mkv_path = Path(mkv_path)
        self.playback = None
        
    def __enter__(self):
        """Context manager entry point"""
        self.playback = PyK4APlayback(str(self.mkv_path))
        self.playback.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        if self.playback is not None:
            self.playback.close()
            
    def extract_depths(self, output_path=None):
        """
        Extract depth frames from the MKV file and save them as a NumPy array.
        
        Args:
            output_path (str, optional): Path where to save the .npy file. 
                                       If None, will save as "depths.npy" in the same directory as the input.
                                       
        Returns:
            np.ndarray: Array of shape (num_frames, height, width) containing the depth frames
        """
        if output_path is None:
            output_path = self.mkv_path.parent / "depths.npy"
        else:
            output_path = Path(output_path)
            
        # First pass to count frames
        num_frames = 0
        while True:
            try:
                capture = self.playback.get_next_capture()
                if capture.transformed_depth is not None:
                    num_frames += 1
            except EOFError:
                break
                
        # Reset playback
        self.playback.close()
        self.playback.open()
        
        # Get dimensions from first frame
        first_capture = self.playback.get_next_capture()
        height, width = first_capture.transformed_depth.shape
        
        # Reset again
        self.playback.close()
        self.playback.open()
        
        # Second pass to collect depths
        depths = np.zeros((num_frames, height, width), dtype=np.uint16)
        
        with tqdm(total=num_frames, desc="Extracting depth frames") as pbar:
            frame_idx = 0
            while True:
                try:
                    capture = self.playback.get_next_capture()
                    depth_frame = capture.transformed_depth
                    
                    if depth_frame is not None:
                        depths[frame_idx] = depth_frame
                        frame_idx += 1
                        pbar.update(1)
                        
                except EOFError:
                    break
                    
        # Convert depths from millimeters to meters
        depths = depths.astype(np.float32) / 1000.0
                    
        # Save the array
        np.save(str(output_path), depths)
        print(f"Saved depth array of shape {depths.shape} to {output_path} (in meters)")
        
        return depths

def main():
    parser = argparse.ArgumentParser(description='Extract depth data from MKV file to NumPy array')
    parser.add_argument('--mkv_path', type=str, required=True, help='Path to input MKV file')
    parser.add_argument('--output_path', type=str, help='Path for output NPY file (optional)')
    args = parser.parse_args()
    
    with MKVDepthExtractor(args.mkv_path) as extractor:
        extractor.extract_depths(args.output_path)

if __name__ == '__main__':
    main()
