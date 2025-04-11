import os
import gc
import argparse
import torch
from safetensors.torch import load_file
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import decord
from PIL import Image
import numpy as np
import torch.nn as nn

# Assuming DepthAnythingV2 is available; replace with actual import if different
from utils.depth_anything_v2.dpt import DepthAnythingV2

### Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo LoRA test script with depth control support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="./models",
        help="Path to pretrained model base directory",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA file to test",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=128,
        help="LoRA alpha, defaults to 128",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test/test_lora",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for inference",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width for inference",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height for inference",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=33,
        help="Number of frames per video, must be divisible by 4+1",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        help="Number of steps for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person typing on a laptop keyboard",
        help="Prompt for inference",
    )
    parser.add_argument(
        "--control_video",
        type=str,
        default=None,
        help="Path to control video for depth control LoRA",
    )
    parser.add_argument(
        "--depth_model_path",
        type=str,
        default="./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
        help="Path to DepthAnythingV2 model checkpoint",
    )
    args = parser.parse_args()
    return args

### Transformer Wrapper for Depth Control
class ControlTransformerWrapper(nn.Module):
    def __init__(self, transformer, control_latents):
        super().__init__()
        self.transformer = transformer
        self.control_latents = control_latents.to(transformer.device, transformer.dtype)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance, attention_kwargs=None, return_dict=False):
        input_hidden_states = torch.cat([hidden_states, self.control_latents], dim=1)
        return self.transformer(
            hidden_states=input_hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

### Process Control Video to Generate Control Latents
def process_control_video(video_path, height, width, num_frames, vae, device, depth_model_path):
    # Load depth model (adjust based on actual DepthAnythingV2 implementation)
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device)
    depth_model.eval()

    # Load video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # [num_frames, H, W, 3]

    # Resize frames
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
        resized_frames.append(np.array(img))
    video_array = np.stack(resized_frames)  # [num_frames, height, width, 3]

    # Compute depth maps
    depth_maps = []
    for frame in video_array:
        depth = depth_model.infer_image(frame)  # [H, W], adjust based on actual API
        depth_maps.append(depth)
    depth_array = np.stack(depth_maps)  # [num_frames, height, width]

    # Normalize depth maps to [-1, 1]
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    if depth_max > depth_min:
        depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 2 - 1
    else:
        depth_array = np.zeros_like(depth_array) - 1

    # Convert to 3 channels for VAE
    depth_array = np.stack([depth_array] * 3, axis=-1)  # [num_frames, height, width, 3]
    depth_tensor = torch.from_numpy(depth_array).permute(0, 3, 1, 2).float().to(device)  # [num_frames, 3, height, width]

    # Encode to latents
    with torch.no_grad():
        latents = vae.encode(depth_tensor.unsqueeze(0)).latent_dist.sample() * vae.config.scaling_factor
    return latents  # [1, C, F, H, W]

### Main Inference Function
@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load base pipeline
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.pretrained_model, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.enable_sequential_cpu_offload()

    # Base inference
    print("Running base model inference...")
    output_base = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).frames[0]
    export_to_video(output_base, os.path.join(args.output_dir, "output_base.mp4"), fps=15)
    print("Base model inference completed.")

    # LoRA inference
    if args.lora:
        print("Running LoRA inference...")
        # Clean up previous transformer
        del transformer
        pipe.transformer = None
        gc.collect()
        torch.cuda.empty_cache()

        # Load new transformer
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(device)

        control_latents = None
        if args.control_video:
            # Modify transformer for depth control
            with torch.no_grad():
                old_proj = transformer.x_embedder.proj
                old_in_channels = transformer.config.in_channels
                new_in_channels = old_in_channels * 2
                new_proj = nn.Conv3d(
                    new_in_channels,
                    old_proj.out_channels,
                    kernel_size=old_proj.kernel_size,
                    stride=old_proj.stride,
                    padding=old_proj.padding,
                ).to(device, torch.bfloat16)
                new_proj.weight.zero_()
                new_proj.bias.zero_()
                new_proj.weight.data[:, :old_in_channels] = old_proj.weight.data
                new_proj.bias.data = old_proj.bias.data
                transformer.x_embedder.proj = new_proj
                transformer.register_to_config(in_channels=new_in_channels)

            # Process control video
            print("Processing control video for depth control...")
            control_latents = process_control_video(
                args.control_video, args.height, args.width, args.num_frames, pipe.vae, device, args.depth_model_path
            )

        # Load LoRA weights
        lora_sd = load_file(args.lora)
        rank = next(iter(lora_sd.values())).shape[0]
        alpha = args.alpha
        lora_weight = alpha / rank
        print(f"LoRA rank={rank}, alpha={alpha}, lora_weight={lora_weight}")
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        transformer.set_adapters(adapter_names="default_lora", weights=lora_weight)

        # Set transformer in pipeline
        if control_latents is not None:
            pipe.transformer = ControlTransformerWrapper(transformer, control_latents)
        else:
            pipe.transformer = transformer

        # Run LoRA inference
        output_lora = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        ).frames[0]
        export_to_video(output_lora, os.path.join(args.output_dir, "output_lora.mp4"), fps=15)
        print("LoRA inference completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)