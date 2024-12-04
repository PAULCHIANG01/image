import torch
import moviepy.editor as mpy
import cv2
import numpy as np
from models.frame_synthesis import *
from utils_test_frame_synthesis import *
# from utils.flow import optical_flow_raft, get_raft_model
from utils_test_frame_synthesis.dataset import FrameFlowProcessing


# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL_PATH = "data/models/frame_synthesis.pth"
model = Synthesis().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()
transforms = FrameFlowProcessing()
print(f"Model loaded from {MODEL_PATH}")

# Video parameters
video_path = "data/videos/Lion_18.mp4"
start_sec = 0
fps = 30
num_frames = 120

def compute_opencv_flow(src_frame, tgt_frame):
    """
    Compute optical flow using OpenCV's Farneback method.
    :param src_frame: Source frame (HxWx3, uint8).
    :param tgt_frame: Target frame (HxWx3, uint8).
    :return: Optical flow (HxWx2, float32).
    """
    # Ensure 3 channels
    if len(src_frame.shape) == 2 or src_frame.shape[-1] == 1:
        src_frame = cv2.cvtColor(src_frame, cv2.COLOR_GRAY2BGR)
    if len(tgt_frame.shape) == 2 or tgt_frame.shape[-1] == 1:
        tgt_frame = cv2.cvtColor(tgt_frame, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale
    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        src_gray, tgt_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


frames = get_frames(video_path, w=256, h=160, start_sec=start_sec, fps=fps, f=num_frames)

# Ensure frames are loaded correctly
if frames is None or len(frames) == 0:
    raise ValueError("No frames were extracted from the video. Check the video path or parameters.")

# Process frames
processed_frames = []
for i in range(len(frames) - 1):
    # Preprocess frames: Convert frames to tensors
    src_frame_tensor = transforms.process_frame(frames[i])  # Tensor
    next_frame_tensor = transforms.process_frame(frames[i + 1])  # Tensor

    # Convert tensors back to NumPy arrays for OpenCV
    src_frame_np = transforms.deprocess_frame(src_frame_tensor)
    next_frame_np = transforms.deprocess_frame(next_frame_tensor)

    # Compute optical flow using OpenCV
    flow_np = compute_opencv_flow(src_frame_np, next_frame_np)

    # Convert optical flow to PyTorch tensor
    flow = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Convert source frame to tensor and add batch dimension
    src_frame = src_frame_tensor.unsqueeze(0).to(DEVICE)

    # Predict the next frame using the model
    with torch.no_grad():
        predicted_frame = model(src_frame, flow)

    # Deprocess predicted frame and add to processed frames
    processed_frame = transforms.deprocess_frame(predicted_frame.squeeze(0))  # NumPy array
    processed_frames.append(processed_frame)

# Save the processed video
output_clip = mpy.ImageSequenceClip(processed_frames, fps=fps)
output_clip.write_videofile("output_video.mp4", codec="libx264", logger=None)
