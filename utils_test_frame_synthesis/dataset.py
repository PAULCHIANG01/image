import os
import torch
import random
import numpy as np
from tqdm import tqdm
from .utils import *


class FrameSpectrumProcessing:
    def __init__(self, num_freq, fft=True, scale=2.82, std_path="data/labels/fft_std.npy"):
        self.num_freq = num_freq
        self.fft = fft
        self.scale = scale
        self.std = load_npy(std_path)[:, None, None, None]
        self.num_channels = 4 if fft else 2
        self.num_freq_total = self.std.shape[0]

    def process_spec(self, spec):
        spec = spec[:self.num_freq] / (self.std[:self.num_freq] * self.scale)
        return torch.from_numpy(spec).permute(0, 3, 1, 2)

    def deprocess_spec(self, spec):
        spec = spec.permute(0, 2, 3, 1).numpy()
        spec *= self.std[:spec.shape[1]] * self.scale
        flow = spec_to_flow(pad_spectrum(spec, self.num_freq_total, fft=self.fft), fft=self.fft)
        return flow.astype(DTYPE_NUMPY)

class SpectrumDataset(torch.utils.data.Dataset, FrameSpectrumProcessing):
    """Dataset for motion synthesis VAE."""
    
    def __init__(
        self, 
        num_freq: int,
        is_train: bool, 
        fft: bool = True,
        scale: float = 2.82,
        std_path: str = "data/labels/fft_std.npy",
        label_dir: str = "data/labels",
        video_dir: str = "data/videos",
        flow_dir: str = "data/flow",
    ):
        super().__init__(num_freq, fft, scale, std_path)
        
        self.is_train = is_train
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        self.label_dir = label_dir
        
        with open(os.path.join(label_dir, f"motion_synthesis_{'train' if is_train else 'test'}_set.csv")) as f:
            self.data = {}
            num_seqs = 0
            
            f.readline()
            for line in f:
                video_id, start_sec, num_frames, fps = line.strip().split(",")
                start_sec, num_frames, fps = int(start_sec), int(num_frames), float(fps)
                flow_path = os.path.join(flow_dir, f"{video_id}_{start_sec:03d}.npy")
                if os.path.exists(flow_path):
                    if video_id not in self.data:
                        self.data[video_id] = []
                    self.data[video_id].append({"start_sec": start_sec, "fps": fps})
                    num_seqs += 1
            self.data = list(self.data.items())
        
        print(f"{type(self).__name__} ({'train' if self.is_train else 'test'}): {len(self.data)} videos, {num_seqs} sequences")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id, sequences = self.data[idx]
        if self.is_train:
            seq = random.choice(sequences)
        else:
            seq = sequences[0]
        spec = self._get_spec(video_id, seq["start_sec"])
        if self.is_train:
            i = random.choice(range(self.num_freq))
            return spec[i]  # shape (num_channels, height, width)
        else:
            return spec  # shape (num_frequencies, num_channels, height, width)
    
    def _get_spec(self, video_id, start_sec):
        flow = load_npy(os.path.join(self.flow_dir, f"{video_id}_{start_sec:03d}.npy"))
        spec = flow_to_spec(flow, fft=self.fft)
        return self.process_spec(spec)
    
    def get_std_from_zero(self, max_seqs_per_vid=1):
        """Compute standard deviation from zero of motion spectrums, which is needed to normalize input data to the diffusion model."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        std = np.zeros((149,), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(std)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)
                m += np.square(spec).mean(axis=(1, 2, 3))
                n += 1
                if max_seqs_per_vid is not None and n >= max_seqs_per_vid:
                    break
            std += m / n
        std = np.sqrt(std / len(self.data))
        
        plt.plot(std)
        plt.xlabel("Frequency index")
        plt.ylabel("Standard deviation")
        plt.title(f"{'FFT' if self.fft else 'DCT'} standard deviation from zero")
        plt.show()
        plt.close()
        
        return std
    
    def test_scales(self, std, min_scale=1, max_scale=4, num_scales=7, max_seqs_per_vid=1):
        """Compute percentage of values in spectrums that is out of the range [-1, 1] with the given `std` and different values of `scale`."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        std = std[:self.num_freq, None, None, None]
        scales = np.linspace(min_scale, max_scale, num_scales)

        out_of_range = np.zeros((self.num_freq, num_scales), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(out_of_range)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)[:self.num_freq] / std
                spec = np.abs(spec)
                for i, s in enumerate(scales):
                    m[:, i] += ((spec / s) > 1).astype(spec.dtype).mean(axis=(1, 2, 3))
                n += 1
                if max_seqs_per_vid is not None and n >= max_seqs_per_vid:
                    break
            out_of_range += m / n
        out_of_range /= len(self.data)
        out_of_range *= 100
        
        for i in range(self.num_freq):
            plt.plot(scales, out_of_range[i, :], label=str(i), ls="-", alpha=.3)
        plt.plot(scales, out_of_range.mean(axis=0), label="mean", ls="-", marker="^")
        plt.xlabel("Scale")
        plt.ylabel("Percentage (%)")
        plt.title(f"Out of range values in {'FFT' if self.fft else 'DCT'}")
        plt.legend()
        
        return scales, out_of_range#.mean(axis=0)
    
    def reconstruct_flow(self, num_freqs=[1, 2, 4, 8, 16, 32, 64, 75]):
        """Compute mean squared errors of reconstructing optical flow from spectrums with limited numbers of frequencies."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        mse = np.zeros(len(num_freqs), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(mse)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)
                for i, num_freq in enumerate(num_freqs):
                    flow_ = spec_to_flow(pad_spectrum(truncate_spectrum(spec, num_freq, fft=self.fft), 149, fft=self.fft), fft=self.fft)
                    m[i] += np.sqrt(np.square(flow - flow_).mean())
                n += 1
                
                break
            
            mse += m / n
        mse /= len(self.data)
        
        plt.plot(num_freqs, mse)
        plt.xlabel("Number of frequences")
        plt.ylabel("Mean squared error")
        plt.title(f"Reconstructing optical flow from truncated {'FFT' if self.fft else 'DCT'}")
        plt.show()
        plt.close()
        
        return num_freqs, mse

class FrameSpectrumDataset(SpectrumDataset):
    """Dataset for motion synthesis U-Net."""
    
    def __getitem__(self, idx):
        video_id, sequences = self.data[idx]
        if self.is_train:
            seq = random.choice(sequences)
        else:
            seq = sequences[0]
        
        # motion spectrum
        spec = self._get_spec(video_id, seq["start_sec"])
        _, _, h, w = spec.shape
        if self.is_train:
            freq_idx = torch.randint(0, self.num_freq, tuple(), dtype=torch.long)
            spec = spec[freq_idx]
        else:
            freq_idx = torch.arange(self.num_freq, dtype=torch.long)
        
        # first frame
        frame = get_frames(os.path.join(self.video_dir, f"{video_id}.mp4"), w, h, seq["start_sec"], f=1, fps=seq["fps"])[0]

        return (self.process_frame(frame), freq_idx, spec)


DTYPE_NUMPY = np.float32
DTYPE_TORCH = torch.float32

class FrameFlowProcessing:
    """Handles preprocessing and deprocessing of frames and optical flows."""

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=DTYPE_TORCH)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=DTYPE_TORCH)

    def __init__(self, frame_h=160, frame_w=256):
        self.frame_h = frame_h
        self.frame_w = frame_w

    def process_frame(self, frame):
        """Normalize a single frame or a batch of frames."""
        ndims = len(frame.shape)
        assert ndims in [3, 4], frame.shape
        frame = torch.from_numpy(frame.astype(DTYPE_NUMPY)) / 255
        if ndims == 3:  # Single frame
            return (frame.permute(2, 0, 1) - self.mean[:, None, None]) / self.std[:, None, None]
        else:  # Multiple frames
            return (frame.permute(0, 3, 1, 2) - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def denormalize_frame(self, frame):
        """De-normalize a batch of frames."""
        assert len(frame.shape) == 4, frame.shape
        return torch.clip(
            frame * self.std[None, :, None, None].to(frame.device)
            + self.mean[None, :, None, None].to(frame.device),
            0,
            1,
        )

    def deprocess_frame(self, frame):
        """
        Convert a tensor frame to a NumPy array and ensure it's RGB.
        """
        assert len(frame.shape) in [3, 4], f"Expected 3D or 4D tensor, got shape {frame.shape}"

        if len(frame.shape) == 3:
            frame = frame.cpu() * self.std[:, None, None] + self.mean[:, None, None]
            frame = frame.permute(1, 2, 0).numpy() * 255
        else:
            frame = frame.cpu() * self.std[None, :, None, None] + self.mean[None, :, None, None]
            frame = frame.permute(0, 2, 3, 1).numpy() * 255

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Ensure 3 channels (convert grayscale to RGB if necessary)
        if len(frame.shape) == 3 and frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 2:  # Grayscale without explicit channel dim
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame

    def process_flow(self, flow):
        """Process optical flow."""
        ndims = len(flow.shape)
        assert ndims in [3, 4], flow.shape
        flow = torch.from_numpy(flow)
        if ndims == 3:  # Single flow field
            return flow.permute(2, 0, 1)
        else:  # Multiple flow fields
            return flow.permute(0, 3, 1, 2)

    def deprocess_flow(self, flow):
        """Deprocess optical flow back to NumPy."""
        assert len(flow.shape) == 4, flow.shape
        return flow.cpu().permute(0, 2, 3, 1).numpy()


class FrameFlowDataset(torch.utils.data.Dataset, FrameFlowProcessing):
    """Dataset for frame synthesis model."""
    
    def __init__(
        self, 
        is_train: bool,
        frame_h: int = 160, 
        frame_w: int = 256,  
        label_dir: str = "data/labels",
        video_dir: str = "data/videos",
        flow_dir: str = "data/flow",
    ):
        super().__init__(frame_h, frame_w)
        
        self.is_train = is_train
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        
        with open(os.path.join(label_dir, f"frame_synthesis_{'train' if is_train else 'test'}_set.csv")) as f:
            self.data = []
            f.readline()
            for line in f:
                video_id, start_sec, num_frames, fps = line.strip().split(",")
                start_sec, num_frames, fps = int(start_sec), int(num_frames), float(fps)
                flow_path = os.path.join(flow_dir, f"{video_id}_{start_sec:03d}.npy")
                assert os.path.exists(flow_path), flow_path
                self.data.append((video_id, start_sec, num_frames, fps))
        
        print(f"{type(self).__name__} ({'train' if self.is_train else 'test'}): {len(self.data)} sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id, start_sec, num_frames, fps = self.data[idx]
        
        frames = self.get_frames(os.path.join(self.video_dir, f"{video_id}.mp4"), start_sec, num_frames, fps)
        flow = load_npy(os.path.join(self.flow_dir, f"{video_id}_{start_sec:03d}.npy"))
        
        if self.is_train:
            t = random.choice(range(1, num_frames))
            return (
                self.process_frame(frames[0]),   # source frame, shape (3, height, width)
                self.process_frame(frames[t]),   # target frame, shape (3, height, width)
                self.process_flow(flow[t - 1]),  # optical flow, shape (2, height, width)
            )
        else:
            return (
                self.process_frame(frames[0]),   # first frame, shape (3, height, width)
                self.process_frame(frames[1:]),  # other frames, shape (num_frames, 3, height, width)
                self.process_flow(flow),         # optical flow, shape (num_frames, 2, height, width)
            )
