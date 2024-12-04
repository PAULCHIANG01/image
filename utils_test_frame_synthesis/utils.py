import cv2
import json
import torch
import scipy
import subprocess
import numpy as np
from typing import Union
from PIL import Image
import moviepy.editor as mpy

from .flow_vis import make_colorwheel

DTYPE_NUMPY = np.float32
DTYPE_TORCH = torch.float32

# Replace pyflow's optical flow with OpenCV's Farneback Optical Flow
def optical_flow(src_frame: np.ndarray, tgt_frame: np.ndarray):
    """
    Compute optical flow using OpenCV's Farneback method.
    :param src_frame: Source frame (HxWx3, uint8).
    :param tgt_frame: Target frame (HxWx3, uint8).
    :return: Optical flow (HxWx2, float32).
    """
    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        src_gray, tgt_gray, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

# Frame extraction function
def get_frames(inp: str, w: int, h: int, start_sec: float = 0, t: float = None, f: int = None, fps=None) -> np.ndarray:
    """
    Extract frames from a video using ffmpeg.
    :param inp: Path to the input video.
    :param w: Width of the extracted frames.
    :param h: Height of the extracted frames.
    :param start_sec: Start time in seconds.
    :param t: Duration to extract in seconds.
    :param f: Number of frames to extract.
    :param fps: Frames per second for extraction.
    :return: A NumPy array of shape (num_frames, h, w, 3).
    """
    args = []
    if t is not None:
        args += ["-t", f"{t:.2f}"]
    elif f is not None:
        args += ["-frames:v", str(f)]
    if fps is not None:
        args += ["-r", str(fps)]

    args = [
        "ffmpeg", "-nostdin", "-ss", f"{start_sec:.2f}", "-i", inp, *args,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:"
    ]

    process = subprocess.Popen(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{inp}: ffmpeg error: {err.decode('utf-8')}")

    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)

# Update all other functionalities
def flow_to_spec(flow: np.ndarray, fft: bool = True):
    assert len(flow.shape) == 4, flow.shape
    if fft:
        spec = np.fft.fft(flow, axis=0)
        return np.concatenate([spec.real, spec.imag], axis=-1)
    else:
        return scipy.fft.dct(flow, axis=0, norm="ortho")

def spec_to_flow(spec: np.ndarray, fft: bool = True):
    ndims = len(spec.shape)
    assert ndims in [4, 5], spec.shape
    axis = 0 if ndims == 4 else 1
    if fft:
        assert spec.shape[-1] == 4, spec.shape
        return np.fft.ifft(spec[..., :2] + spec[..., 2:] * 1j, axis=axis)
    else:
        assert spec.shape[-1] == 2, spec.shape
        return scipy.fft.idct(spec, axis=axis, norm="ortho")

# Other utility functions remain unchanged
