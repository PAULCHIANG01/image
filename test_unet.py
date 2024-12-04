import os
import torch
from diffusers import UNet2DModel
import datetime
from utils import *
from models.motion_synthesis import *

# 配置参数

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
_MODEL_DIR = "./data/models"
MODEL_NAME = "unet_v2"
_NAME = "unet_v2"
CKPT_PATH = os.path.join(_MODEL_DIR, f"{MODEL_NAME}.pth")
OUT_DIR = os.path.join(_MODEL_DIR, _NAME + "_samples")
BATCH_SIZE = 1
NUM_WORKERS = 0

FFT = True
NUM_FREQ = 16
SPEC_CHANNELS = 4 if FFT else 2
FRAME_CHANNELS = 3
_VAE_LATENT_CHANNELS = 3
SPEC_LATENT_CHANNELS = SPEC_CHANNELS * _VAE_LATENT_CHANNELS
FRAME_LATENT_CHANNELS = _VAE_LATENT_CHANNELS
LATENT_HEIGHT = 40
LATENT_WIDTH = 64
HEIGHT = LATENT_HEIGHT * 4
WIDTH = LATENT_WIDTH * 4
train_loader = torch.utils.data.DataLoader(FrameSpectrumDataset(NUM_FREQ, is_train=True, fft=FFT), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# 定义模型
vae = get_pretrained_vae().to(DEVICE).eval()
noise_scheduler = get_noise_scheduler()
model = UNet2DModel(**{
    "in_channels": SPEC_LATENT_CHANNELS + FRAME_LATENT_CHANNELS,
    "out_channels": SPEC_LATENT_CHANNELS,
    "class_embed_type": "timestep",
}).to(DEVICE)



# 加载检查点
if os.path.exists(CKPT_PATH):
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from {CKPT_PATH}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

# 进行推理（测试）
@torch.no_grad()
def inference(model, test_ids=None, num_freq=16, num_steps=100, batch_size=1):
    model.eval()

    if test_ids is None:
        # test_ids = ["512px-Dandelion_picture.jpg"]
        test_ids = ["512px-Autumn_leaf_in_the_wind_(Unsplash).jpg"]
    elif isinstance(test_ids, str):
        test_ids = [test_ids]

    for tid in test_ids:
        frame_np = get_image(f"data/images/{tid}", WIDTH, HEIGHT, crop=True)
        frame = train_loader.dataset.process_frame(frame_np).unsqueeze(0).to(DEVICE)

        spec_np = generate_spectrum(vae, model, noise_scheduler, frame, num_freq=num_freq, num_steps=num_steps, batch_size=batch_size)

        spec_image, video = visualize_sample(frame_np, spec_np, train_loader.dataset, magnification=2.0, include_flow=True, fps=30)

        ts = datetime.datetime.now().isoformat().replace(":", "_")
        spec_image.save(os.path.join(OUT_DIR, f"{tid}_{ts}_ddpm{num_steps}_spec.png"))
        video.write_videofile(os.path.join(OUT_DIR, f"{tid}_{ts}_ddpm{num_steps}_flow.mp4"), logger=None)

# 调用推理函数
# inference(model, test_ids="512px-Dandelion_picture.jpg")
inference(model, test_ids="512px-Autumn_leaf_in_the_wind_(Unsplash).jpg")