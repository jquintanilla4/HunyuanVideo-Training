import random
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, InterpolationMode
from PIL import Image
from utils.depth_anything_v2.dpt import DepthAnythingV2
from transformers import BitsAndBytesConfig
import torch.nn as nn
import numpy as np
import decord
from diffusers.utils import export_to_video
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from safetensors.torch import load_file
import torch
import argparse
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo LoRA test script with depth control support")
    parser.add_argument("--pretrained_model", type=str,
                        default="./models",
                        help="Path to pretrained model base directory")
    parser.add_argument("--lora", type=str,
                        default=None,
                        help="LoRA file to test")
    parser.add_argument("--alpha", type=int,
                        default=128,
                        help="LoRA alpha, defaults to 128")
    parser.add_argument("--lora_weight", type=float,
                        default=None,
                        help="Override the computed LoRA weight if provided")
    parser.add_argument("--guidance_scale", type=float,
                        default=6.0,
                        help="Guidance scale for LoRA inference, aka CFG")
    parser.add_argument("--control_scale", type=float,
                        default=None,
                        help="Max dynamic scale for control latents (uses schedule if set)")
    parser.add_argument("--constant", type=float,
                        default=None,
                        help="Constant scale for control latents (overrides dynamic)")
    parser.add_argument("--output_dir", type=str,
                        default="./test/test_lora",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Seed for inference")
    parser.add_argument("--width", type=int,
                        default=512,
                        help="Width for inference")
    parser.add_argument("--height", type=int,
                        default=512,
                        help="Height for inference")
    parser.add_argument("--num_frames", type=int,
                        default=33,
                        help="Number of frames per video, must be divisible by 4+1")  # math term: N ≡ 1 (mod 4); formula: n = 4k + 1
    parser.add_argument("--inference_steps", type=int,
                        default=20,
                        help="Number of steps for inference")
    parser.add_argument("--prompt", type=str,
                        default="A person typing on a laptop keyboard",
                        help="Prompt for inference")
    parser.add_argument("--control_video", type=str,
                        default=None,
                        help="Path to control video for depth control LoRA")
    parser.add_argument("--depth_model_path", type=str,
                        default="./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
                        help="Path to DepthAnythingV2 model checkpoint")
    parser.add_argument("--skip_base_inference",
                        action="store_true",
                        help="Skip the base model inference step")
    # Flag to swap channels if needed (e.g., BGR -> RGB).
    parser.add_argument("--swap_channels",
                        action="store_true",
                        help="Swap color channels (BGR->RGB) if necessary")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for output videos")
    return parser.parse_args()


### Process Control Video to Generate Control Latents
def process_control_video(video_path, height, width, num_frames, vae, device, depth_model_path, output_dir, fps):
    # Load DepthAnythingV2 model
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()
    depth_model.requires_grad_(False)

    # Load control video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # [num_frames, H, W, 3]

    # Resize frames
    resized_frames = [Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS) for frame in frames]
    video_array = np.stack([np.array(img) for img in resized_frames])  # [num_frames, H, W, 3]

    # Convert to tensor and normalize to [-1, 1] for VAE input compatibility (as done in training datasets)
    pixels = torch.from_numpy(video_array).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1.0
    pixels = pixels.unsqueeze(0)        # [1, num_frames, 3, H, W]
    pixels = pixels.permute(0, 2, 1, 3, 4)  # [1, 3, num_frames, H, W]

    # Process depth maps (aligned with training script logic)
    B, C, F, H, W = pixels.shape
    depth_tensor = torch.zeros((B, F, H, W), device=device, dtype=torch.float32)

    for b in range(B):
        for f in range(F):
            # Extract and de-normalize frame to [0,1] for depth model
            frame = (pixels[b, :, f].float() * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
            depth = depth_model.infer_image(frame)  # returns numpy or tensor

            # Ensure tensor on device
            if not isinstance(depth, torch.Tensor):
                depth = torch.from_numpy(depth)
            depth = depth.to(pixels.device, dtype=torch.float32)

            # Resize if shape mismatch
            if depth.shape != (H, W):
                depth = resize(depth.unsqueeze(0).unsqueeze(0),
                               (H, W),
                               interpolation=InterpolationMode.BICUBIC
                               ).squeeze()

            # Normalize to [0,1]
            mi, ma = depth.min(), depth.max()
            if ma > mi:
                depth = (depth - mi) / (ma - mi)

            # Shift to [-1,1]
            depth_tensor[b, f] = depth * 2.0 - 1.0

    # Save depth maps as a video for inspection
    print("Saving depth maps as a video for inspection...")
    depth_visual = (-depth_tensor + 1.0) / 2.0  # [0,1] with 1 being close, 0 being far
    depth_display = (depth_visual * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)  # [B, F, H, W]
    if depth_display.shape[0] == 1:
        depth_display = depth_display.squeeze(0)  # [F, H, W]
    else:
        print(f"Warning: Batch size {depth_display.shape[0]} > 1, saving only the first item's depth.")
        depth_display = depth_display[0]

    # Ensure it's 3 channels for video export
    if depth_display.ndim == 3:  # [F, H, W]
        depth_frames_display = np.stack([depth_display] * 3, axis=-1)  # [F, H, W, 3]
    else:
        raise ValueError(f"Unexpected depth_display dimensions: {depth_display.shape}")

    depth_video_path = os.path.join(output_dir, "depth_video.mp4")
    export_to_video(depth_frames_display, depth_video_path, fps=fps)
    print(f"Depth video saved to {depth_video_path}")

    # Create control latents: replicate depth map to 3 channels.
    depth_tensor = depth_tensor.unsqueeze(1)  # [B, 1, F, H, W]
    control = depth_tensor.repeat(1, 3, 1, 1, 1).to(dtype=vae.dtype)

    # Check shape consistency and range
    assert control.shape[0] == pixels.shape[0]
    assert control.shape[2] == pixels.shape[2] # F
    assert control.shape[3] == pixels.shape[3] # H
    assert control.shape[4] == pixels.shape[4] # W
    assert control.shape[1] == 3 # Channels
    assert not torch.isnan(control).any()
    # Use a small epsilon for float comparisons
    assert (control >= -1.0 - 1e-6).all() and (control <= 1.0 + 1e-6).all(), f"Control range error: min={control.min()}, max={control.max()}"
    # Verify that the depth map preprocessing resulted in values within the expected [-1, 1] bounds

    with torch.no_grad():
        latents = vae.encode(control).latent_dist.sample() * vae.config.scaling_factor

    del depth_model, control, pixels, depth_tensor # Clean up tensors
    gc.collect()
    torch.cuda.empty_cache()
    return latents  # [1, control_channels, F, H', W']


def get_control_scale(t, total_timesteps, max_scale=2.0):
    """
    Compute the control scale for a given timestep during the denoising process.

    Args:
        t (int or float): The current timestep (should be in [0, total_timesteps]).
        total_timesteps (int or float): The total number of timesteps in the denoising process.
        max_scale (float, optional): The maximum control scale to apply at timestep 0. Default is 2.0.

    Returns:
        float: The control scale for the current timestep, linearly decreasing from max_scale to 0.
    """
    scale = max_scale * (1 - t / total_timesteps)
    return scale


def get_reverse_control_scale(t, total_timesteps, max_scale=2.0):
    """
    Compute the reverse control scale for a given timestep during the denoising process.

    Args:
        t (int or float): The current timestep (should be in [0, total_timesteps]).
        total_timesteps (int or float): The total number of timesteps in the denoising process.
        max_scale (float, optional): The maximum control scale to apply at timestep 0. Default is 2.0.

    Returns:
        float: The reverse control scale for the current timestep, linearly increasing from 0 to max_scale.
    """
    reverse_scale = max_scale * (t / total_timesteps) # scale = 2.0 * (i / total_timesteps)
    return reverse_scale


### Main Inference Function
@torch.inference_mode()
def main(args):
    def set_all_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # ensure full reproducibility
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip base inference if requested.
    if args.skip_base_inference:
        print("Skipping base model inference.")
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.pretrained_model, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = HunyuanVideoPipeline.from_pretrained(args.pretrained_model, transformer=transformer, torch_dtype=torch.float16)
        pipe.vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_min_num_frames=64,
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
            tile_sample_stride_num_frames=16,
        )
        pipe.enable_sequential_cpu_offload()
        print("Running base model inference...")
        output_base = pipe(
            prompt=args.prompt, height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.inference_steps, generator=torch.Generator(device=device).manual_seed(args.seed)
        ).frames[0]

        export_to_video(output_base, os.path.join(args.output_dir, "output_base.mp4"), fps=args.fps)
        print("Base model inference completed.")
        del output_base, transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()

    # LoRA Inference
    if args.lora:
        print("Running LoRA inference...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16, bnb_8bit_use_double_quant=True)
        print("Loading and quantizing transformer...")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.pretrained_model,
                                                                     subfolder="transformer",
                                                                     quantization_config=quant_config,
                                                                     torch_dtype=torch.bfloat16)

        print(f"After loading transformer: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
              f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        control_latents = None
        original_in_channels = transformer.config.in_channels  # e.g. 16

        if args.control_video:
            from diffusers import AutoencoderKLHunyuanVideo
            vae = AutoencoderKLHunyuanVideo.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
            control_latents = process_control_video(args.control_video, args.height, args.width, args.num_frames, vae, device, args.depth_model_path, args.output_dir, args.fps)
            del vae
            gc.collect()
            torch.cuda.empty_cache()

            lora_sd = load_file(args.lora)
            proj_key = next((key for key in lora_sd.keys() if key.startswith("x_embedder.proj.lora_A")), None)
            if proj_key is not None:
                proj = transformer.x_embedder.proj
                base_proj = proj.base_layer if hasattr(proj, "base_layer") else proj
                old_in_dim = original_in_channels
                new_in_dim = old_in_dim + control_latents.shape[1]  # e.g., 16 + 16 = 32
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

        # Load LoRA Adapter with adjustable weight.
        rank = next(iter(lora_sd.values())).shape[0]
        alpha = args.alpha
        if args.lora_weight is not None:
            lora_weight = args.lora_weight
        else:
            lora_weight = alpha / rank
        print(f"LoRA rank={rank}, alpha={alpha}, lora_weight={lora_weight}")
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        transformer.set_adapters(adapter_names="default_lora", weights=lora_weight)

        pipe = HunyuanVideoPipeline.from_pretrained(args.pretrained_model, transformer=transformer, torch_dtype=torch.float16)
        pipe.vae.enable_tiling(
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_min_num_frames=16,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
            tile_sample_stride_num_frames=8,
        )
        pipe.enable_sequential_cpu_offload()

        # disable dropout/randomness
        transformer.eval()
        pipe.vae.eval()

        # Custom Pipeline Logic
        batch_size = 1
        num_videos_per_prompt = 1
        generator = torch.Generator(device=device).manual_seed(args.seed)
        guidance_scale = args.guidance_scale

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(prompt=args.prompt,
                                                                                        num_videos_per_prompt=num_videos_per_prompt,
                                                                                        device=device,
                                                                                        max_sequence_length=256)
        transformer_dtype = transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        pipe.scheduler.set_timesteps(args.inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = transformer.config.in_channels  # 16 (or 32 if control latents used)
        num_latent_frames = (args.num_frames - 1) // pipe.vae.temporal_compression_ratio + 1
        height_latent = args.height // pipe.vae.spatial_compression_ratio
        width_latent = args.width // pipe.vae.spatial_compression_ratio
        latents = torch.randn(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_latent_frames,
            height_latent,
            width_latent,
            device=device,
            dtype=torch.float32,
            generator=generator)

        if control_latents is not None:
            control_latents = control_latents.to(latents.device, latents.dtype)
            latents[:, original_in_channels:, :, :, :] = control_latents

        # guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device)

        print("Starting denoising loop...")
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                return_dict=False,
            )[0]

            # Split latents into original and control parts
            original_latents = latents[:, :original_in_channels, :, :, :]
            control_latents_part = latents[:, original_in_channels:, :, :, :]

            # Denoise/update original latents
            updated_latents = pipe.scheduler.step(noise_pred, t, original_latents, return_dict=False)[0]

            # Timestep-dependent control scale
            total_timesteps = len(timesteps)
            if args.control_scale is not None:
                scale = get_control_scale(i, total_timesteps, max_scale=args.control_scale)
            elif args.constant is not None:
                scale = args.constant
            else:
                scale = 1.0
            control_latents_part = control_latents_part * scale

            latents = torch.cat([updated_latents, control_latents_part], dim=1)

        print("Decoding latents with VAE...")
        final_latents = latents[:, :original_in_channels, :, :, :].to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video_tensor = pipe.vae.decode(final_latents, return_dict=False)[0]

        print(f"VAE output shape: {video_tensor.shape}, min: {video_tensor.min()}, max: {video_tensor.max()}")

        ### Post-process for correct colors
        if video_tensor.dim() == 5:
            video_tensor = video_tensor.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
            if video_tensor.shape[0] == 1:
                video_tensor = video_tensor.squeeze(0)  # [F, H, W, C]
            else:
                raise ValueError(f"Batch size > 1 not handled: {video_tensor.shape}")

        # Convert from roughly [-1, 1] to [0, 1]
        video_tensor = (video_tensor + 1.0) / 2.0
        # Permanently flip colors.
        video_tensor = 1.0 - video_tensor

        # Optionally swap channels if needed.
        if args.swap_channels:
            video_tensor = video_tensor[..., [2, 1, 0]]

        # Clamp values to valid range [0,1] before converting to uint8
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        # Scale to [0,255], round to integers, convert to uint8, move to CPU, and convert to numpy
        video_frames = (video_tensor * 255).round().to(torch.uint8).cpu().numpy()
        print(f"Post-processed frames shape: {video_frames.shape}, min: {video_frames.min()}, max: {video_frames.max()}")

        output_path = os.path.join(args.output_dir, "output_lora.mp4")
        export_to_video(video_frames, output_path, fps=args.fps)

        print(f"LoRA inference completed. Video saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)