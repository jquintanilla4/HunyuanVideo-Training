import torch
from safetensors.torch import load_file
import argparse
import numpy as np
from diffusers import HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
import decord
from PIL import Image
from transformers import BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Script to diagnose depth control LoRA issues")
    parser.add_argument("--lora", type=str, required=True, help="Path to the LoRA checkpoint file")
    parser.add_argument("--control_video", type=str, required=True, help="Path to the control video")
    parser.add_argument("--pretrained_model", type=str, default="./models", help="Path to pretrained model directory")
    parser.add_argument("--height", type=int, default=512, help="Height for inference")
    parser.add_argument("--width", type=int, default=512, help="Width for inference")
    parser.add_argument("--num_frames", type=int, default=33, help="Number of frames")
    return parser.parse_args()

def process_control_video(video_path, height, width, num_frames, vae, device):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()
    resized_frames = [Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS) for frame in frames]
    video_array = np.stack([np.array(img) for img in resized_frames])
    pixels = torch.from_numpy(video_array).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    pixels = pixels.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Shape: [1, 3, num_frames, H, W]
    pixels = pixels.to(device, dtype=vae.dtype)  # Cast to torch.bfloat16 and move to device
    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    print(f"control_latents shape: {latents.shape}, min: {latents.min()}, max: {latents.max()}")
    return latents

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LoRA checkpoint
    lora_sd = load_file(args.lora)
    print("Keys in LoRA checkpoint:", list(lora_sd.keys()))

    # Check 1: LoRA Weights
    if "x_embedder.proj.lora_A.weight" in lora_sd and "x_embedder.proj.lora_B.weight" in lora_sd:
        lora_A_sum = lora_sd["x_embedder.proj.lora_A.weight"].abs().sum().item()
        lora_B_sum = lora_sd["x_embedder.proj.lora_B.weight"].abs().sum().item()
        print(f"LoRA Weights - x_embedder.proj.lora_A.weight sum: {lora_A_sum}")
        print(f"LoRA Weights - x_embedder.proj.lora_B.weight sum: {lora_B_sum}")
        if lora_A_sum == 0 or lora_B_sum == 0:
            print("Warning: LoRA weights are zero. Control channels may be ignored.")
    else:
        print("Warning: Projection layer LoRA weights (x_embedder.proj.lora_A/B.weight) not found.")

    # Define quantization config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True
    )

    # Load transformer with quantization (no .to(device))
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16
    )

    # Load VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)

    # Check 2: Control Latents Quality
    control_latents = process_control_video(args.control_video, args.height, args.width, args.num_frames, vae, device)
    if control_latents.min() == control_latents.max():
        print("Warning: Control latents are constant. They won’t influence the output.")

    # Check 3: Projection Layer Update
    original_in_channels = transformer.config.in_channels
    proj_key = "x_embedder.proj.lora_A.weight"
    if proj_key in lora_sd:
        proj = transformer.x_embedder.proj
        new_in_dim = original_in_channels + control_latents.shape[1]
        new_proj = torch.nn.Conv3d(
            new_in_dim, proj.out_channels, kernel_size=proj.kernel_size,
            stride=proj.stride, padding=proj.padding
        ).to(device, torch.bfloat16)
        new_proj.weight.data.zero_()
        new_proj.bias.data.zero_()
        new_proj.weight.data[:, :original_in_channels].copy_(proj.weight.data)
        new_proj.bias.data.copy_(proj.bias.data)
        transformer.x_embedder.proj = new_proj
        transformer.register_to_config(in_channels=new_in_dim)
        print(f"Updated in_channels: {transformer.config.in_channels}")
        if transformer.config.in_channels != 32:
            print("Warning: in_channels is not 32. Control channels may not be processed correctly.")
    else:
        print("Projection layer not updated due to missing LoRA weights.")

if __name__ == "__main__":
    main()