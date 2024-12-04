import torch
import numpy as np
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


def get_raft_model(model_size="small", device=None):
    """
    Load RAFT optical flow model from torchvision.
    """
    if model_size == "small":
        from torchvision.models.optical_flow import raft_small as raft
        from torchvision.models.optical_flow import Raft_Small_Weights as Weights
    else:
        from torchvision.models.optical_flow import raft_large as raft
        from torchvision.models.optical_flow import Raft_Large_Weights as Weights

    weights = Weights.DEFAULT
    transforms = weights.transforms()
    raft_model = raft(weights=weights, progress=False).eval()
    if device is not None:
        raft_model = raft_model.to(device)

    return raft_model, transforms


@torch.no_grad()
def optical_flow_raft(src, tgt, model, transforms, batch_size=1):
    """
    Compute optical flow using RAFT model.
    """
    assert src.dtype == np.uint8 and len(src.shape) == 3, src.shape
    assert tgt.dtype == np.uint8 and len(tgt.shape) in [3, 4], tgt.shape
    assert tgt.shape[-3:] == src.shape, (src.shape, tgt.shape)

    device = next(model.parameters()).device

    if len(tgt.shape) == 3:  # Single target frame
        src = torch.from_numpy(src).unsqueeze(0).permute(0, 3, 1, 2)
        tgt = torch.from_numpy(tgt).unsqueeze(0).permute(0, 3, 1, 2)
        src, tgt = transforms(src, tgt)
        out = model(src.to(device), tgt.to(device))[-1]
        return out.permute(0, 2, 3, 1).cpu().numpy()
    else:  # Batch of target frames
        src = torch.from_numpy(src).unsqueeze(0).permute(0, 3, 1, 2)
        tgt = torch.from_numpy(tgt).permute(0, 3, 1, 2)

        nb = int(np.ceil(tgt.shape[0] / batch_size))
        flow = []
        for i in range(nb):
            s = i * batch_size
            e = min(tgt.shape[0], s + batch_size)
            src_, tgt_ = transforms(src.repeat(e - s, 1, 1, 1), tgt[s:e])
            out = model(src_.to(device), tgt_.to(device))[-1]
            flow.append(out.cpu())
        return torch.cat(flow).permute(0, 2, 3, 1).numpy()


# Example of using the RAFT model directly
if __name__ == "__main__":
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transforms = get_raft_model(model_size="small", device=device)

    # Example input frames
    src_frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Random source frame
    tgt_frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Random target frame

    # Compute optical flow
    flow = optical_flow_raft(src_frame, tgt_frame, model, transforms)
    print("Optical flow shape:", flow.shape)
