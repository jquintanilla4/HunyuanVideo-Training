import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import argparse
import torch
from safetensors.torch import load_file
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import decord
import numpy as np
import torch.nn as nn
from transformers import BitsAndBytesConfig
from utils.depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from torchvision.transforms.functional import resize, InterpolationMode

### Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo LoRA test script with depth control support")
    parser.add_argument("--pretrained_model", type=str, default="./models", help="Path to pretrained model base directory")
    parser.add_argument("--lora", type=str, default=None, help="LoRA file to test")
    parser.add_argument("--alpha", type=int, default=128, help="LoRA alpha, defaults to 128")
    parser.add_argument("--output_dir", type=str, default="./test/test_lora", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Seed for inference")
    parser.add_argument("--width", type=int, default=512, help="Width for inference")
    parser.add_argument("--height", type=int, default=512, help="Height for inference")
    parser.add_argument("--num_frames", type=int, default=33, help="Number of frames per video, must be divisible by 4+1")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of steps for inference")
    parser.add_argument("--prompt", type=str, default="A person typing on a laptop keyboard", help="Prompt for inference")
    parser.add_argument("--control_video", type=str, default=None, help="Path to control video for depth control LoRA")
    parser.add_argument("--depth_model_path", type=str, default="./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth", help="Path to DepthAnythingV2 model checkpoint")
    parser.add_argument("--skip_base_inference", action="store_true", help="Skip the base model inference step")
    return parser.parse_args()

### Process Control Video to Generate Control Latents
def process_control_video(video_path, height, width, num_frames, vae, device, depth_model_path, output_dir):
    # Load DepthAnythingV2 model
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()
    depth_model.requires_grad_(False)

    # Load control video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # Shape: [num_frames, H, W, 3]

    # Resize frames
    resized_frames = [Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS) for frame in frames]
    video_array = np.stack([np.array(img) for img in resized_frames])  # Shape: [num_frames, height, width, 3]

    # Convert to tensor and normalize to [0, 1]
    pixels = torch.from_numpy(video_array).permute(0, 3, 1, 2).to(device).float() / 255.0  # Shape: [num_frames, 3, height, width]
    pixels = pixels.unsqueeze(0)  # Shape: [1, num_frames, 3, height, width]
    pixels = pixels.permute(0, 2, 1, 3, 4)  # Shape: [1, 3, num_frames, height, width]

    # Process depth maps (adapted from training script)
    B, C, F, H, W = pixels.shape
    depth_tensor = torch.zeros((B, F, H, W), device=device)

    for b in range(B):
        for f in range(F):
            frame = pixels[b, :, f].float() * 0.5 + 0.5  # Normalize to [0, 1]
            frame = frame.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]

            depth = depth_model.infer_image(frame)

            if depth.shape != (H, W):
                depth_tensor_tmp = torch.tensor(depth, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                depth_tensor_tmp = resize(depth_tensor_tmp, (H, W), interpolation=InterpolationMode.BICUBIC)
                depth = depth_tensor_tmp.squeeze().cpu().numpy()

            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            depth = depth * 2 - 1  # Scale to [-1, 1]

            depth_tensor[b, f] = torch.tensor(depth, device=device)

    # Save depth maps as a video for inspection
    print("Saving depth maps as a video for inspection...")
    depth_normalized = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * 255
    depth_normalized = depth_normalized.cpu().numpy().astype(np.uint8)  # Shape: [1, num_frames, height, width]
    depth_normalized = depth_normalized.squeeze(0)  # Shape: [num_frames, height, width]
    depth_frames = np.stack([depth_normalized] * 3, axis=-1)  # Shape: [num_frames, height, width, 3]
    depth_video_path = os.path.join(output_dir, "depth_video.mp4")
    export_to_video(depth_frames, depth_video_path, fps=15)
    print(f"Depth video saved to {depth_video_path}")

    # Continue with control tensor creation
    depth_tensor = depth_tensor.unsqueeze(1)  # Shape: [B, 1, F, H, W]
    control = depth_tensor.repeat(1, 3, 1, 1, 1).to(dtype=vae.dtype)  # Shape: [B, 3, F, H, W]

    # Assertions for control tensor
    assert control.shape[0] == pixels.shape[0], f"Batch dimension mismatch: {control.shape[0]} vs {pixels.shape[0]}"
    assert control.shape[2] == pixels.shape[2], f"Frame dimension mismatch: {control.shape[2]} vs {pixels.shape[2]}"
    assert control.shape[3] == pixels.shape[3], f"Height dimension mismatch: {control.shape[3]} vs {pixels.shape[3]}"
    assert control.shape[4] == pixels.shape[4], f"Width dimension mismatch: {control.shape[4]} vs {pixels.shape[4]}"
    assert control.shape[1] == 3, f"Expected control channels to be 3, got {control.shape[1]}"
    assert not torch.isnan(control).any(), "NaN values detected in control tensor"
    assert ((control >= -1.0) & (control <= 1.0)).all(), f"Control values out of range [-1,1]: min={control.min().item()}, max={control.max().item()}"

    # Encode control tensor to latents
    with torch.no_grad():
        latents = vae.encode(control).latent_dist.sample() * vae.config.scaling_factor

    del depth_model, control
    gc.collect()
    torch.cuda.empty_cache()
    return latents  # [1, control_channels, F, H', W']

### Main Inference Function
@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip Base Inference
    if args.skip_base_inference:
        print("Skipping base model inference.")
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.vae.enable_tiling(
            tile_sample_min_height=256, tile_sample_min_width=256, tile_sample_min_num_frames=64,
            tile_sample_stride_height=192, tile_sample_stride_width=192, tile_sample_stride_num_frames=16,
        )
        pipe.enable_sequential_cpu_offload()
        print("Running base model inference...")
        output_base = pipe(
            prompt=args.prompt, height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.inference_steps, generator=torch.Generator(device=device).manual_seed(args.seed),
        ).frames[0]
        export_to_video(output_base, os.path.join(args.output_dir, "output_base.mp4"), fps=15)
        print("Base model inference completed.")
        del output_base, transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()

    # LoRA Inference
    if args.lora:
        print("Running LoRA inference...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16, bnb_8bit_use_double_quant=True
        )

        print("Loading and quantizing transformer...")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", quantization_config=quant_config, torch_dtype=torch.bfloat16
        )
        print(f"After loading transformer: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
              f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        control_latents = None
        original_in_channels = transformer.config.in_channels  # e.g., 16

        # Process Control Video and Update Projection
        if args.control_video:
            from diffusers import AutoencoderKLHunyuanVideo
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                args.pretrained_model, subfolder="vae", torch_dtype=torch.bfloat16
            ).to(device)
            control_latents = process_control_video(
                args.control_video, args.height, args.width, args.num_frames, vae, device, args.depth_model_path, args.output_dir
            )
            del vae
            gc.collect()
            torch.cuda.empty_cache()

            lora_sd = load_file(args.lora)
            proj_key = next((key for key in lora_sd.keys() if key.startswith("x_embedder.proj.lora_A")), None)

            if proj_key is not None:
                proj = transformer.x_embedder.proj
                base_proj = proj.base_layer if hasattr(proj, "base_layer") else proj
                old_in_dim = original_in_channels
                new_in_dim = old_in_dim + control_latents.shape[1]  # e.g., 32

                assert lora_sd[proj_key].shape[1] == new_in_dim, (
                    f"LoRA expects {lora_sd[proj_key].shape[1]} input channels, but new_in_dim is {new_in_dim}"
                )
                in_cls = proj.__class__
                new_proj = in_cls(
                    new_in_dim, base_proj.out_channels, kernel_size=base_proj.kernel_size,
                    stride=base_proj.stride, padding=base_proj.padding,
                ).to(device, torch.bfloat16)
                new_proj.weight.zero_()
                new_proj.bias.zero_()
                new_proj.weight.data[:, :old_in_dim].copy_(base_proj.weight.data)
                new_proj.bias.data.copy_(base_proj.bias.data)
                if hasattr(proj, "base_layer"):
                    proj.base_layer = new_proj
                else:
                    transformer.x_embedder.proj = new_proj
                transformer.register_to_config(in_channels=new_in_dim)
                print(f"Projection layer updated to accept {new_in_dim} channels and config registered.")
            else:
                print("LoRA checkpoint lacks projection key; skipping update.")
                new_in_dim = original_in_channels
        else:
            lora_sd = load_file(args.lora)
            new_in_dim = original_in_channels

        # Load LoRA Adapter
        rank = next(iter(lora_sd.values())).shape[0]
        alpha = args.alpha
        lora_weight = alpha / rank
        print(f"LoRA rank={rank}, alpha={alpha}, lora_weight={lora_weight}")
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        transformer.set_adapters(adapter_names="default_lora", weights=lora_weight)

        # Load Pipeline Components
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.vae.enable_tiling(
            tile_sample_min_height=128, tile_sample_min_width=128, tile_sample_min_num_frames=16,
            tile_sample_stride_height=64, tile_sample_stride_width=64, tile_sample_stride_num_frames=8,
        )
        pipe.enable_sequential_cpu_offload()

        # Custom Pipeline Logic
        batch_size = 1
        num_videos_per_prompt = 1
        generator = torch.Generator(device=device).manual_seed(args.seed)
        guidance_scale = 6.0  # Default from pipeline

        # Encode Prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=args.prompt, num_videos_per_prompt=num_videos_per_prompt, device=device,
            max_sequence_length=256
        )
        transformer_dtype = transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        # Prepare Timesteps
        pipe.scheduler.set_timesteps(args.inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # Prepare Latents
        num_channels_latents = transformer.config.in_channels  # 32 if control is used
        num_latent_frames = (args.num_frames - 1) // pipe.vae.temporal_compression_ratio + 1
        height_latent = args.height // pipe.vae.spatial_compression_ratio
        width_latent = args.width // pipe.vae.spatial_compression_ratio
        latents = torch.randn(
            batch_size * num_videos_per_prompt, num_channels_latents, num_latent_frames, height_latent, width_latent,
            device=device, dtype=torch.float32, generator=generator
        )
        if control_latents is not None:
            control_latents = control_latents.to(latents.device, latents.dtype)
            latents[:, original_in_channels:, :, :, :] = control_latents

        # Guidance Condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # Denoising Loop
        print("Starting denoising loop...")
        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = transformer(
                hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask, pooled_projections=pooled_prompt_embeds,
                guidance=guidance, return_dict=False,
            )[0]

            # Split latents
            original_latents = latents[:, :original_in_channels, :, :, :]
            control_latents_part = latents[:, original_in_channels:, :, :, :]

            # Update only original latents
            updated_latents = pipe.scheduler.step(noise_pred, t, original_latents, return_dict=False)[0]

            # Reconstruct 32-channel latents
            latents = torch.cat([updated_latents, control_latents_part], dim=1)

        # Decode only original latents
        print("Decoding latents with VAE...")
        final_latents = latents[:, :original_in_channels, :, :, :].to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video_tensor = pipe.vae.decode(final_latents, return_dict=False)[0]

        # Debug: Check VAE output
        print(f"VAE output shape: {video_tensor.shape}, min: {video_tensor.min()}, max: {video_tensor.max()}")

        # Post-process video tensor to fix color inversion
        if video_tensor.dim() == 5:
            # Permute to [batch, frames, height, width, channels]
            video_tensor = video_tensor.permute(0, 2, 3, 4, 1)
            if video_tensor.shape[0] == 1:
                video_tensor = video_tensor.squeeze(0)  # [frames, height, width, channels]
            else:
                raise ValueError(f"Batch size > 1 not handled: {video_tensor.shape}")

        # Normalize based on actual range to [0, 1]
        video_min = video_tensor.min()
        video_max = video_tensor.max()
        if video_max > video_min:  # Avoid division by zero
            video_tensor = (video_tensor - video_min) / (video_max - video_min)
        else:
            video_tensor = torch.zeros_like(video_tensor)  # Fallback if range is zero

        # Clamp to [0, 1] and scale to [0, 255]
        video_tensor = torch.clamp(video_tensor, 0, 1)
        video_frames = (video_tensor * 255).to(torch.uint8).cpu().numpy()

        # Debug: Check post-processed frames
        print(f"Post-processed frames shape: {video_frames.shape}, min: {video_frames.min()}, max: {video_frames.max()}")

        # Export to video
        output_path = os.path.join(args.output_dir, "output_lora.mp4")
        export_to_video(video_frames, output_path, fps=15)
        print(f"LoRA inference completed. Video saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)