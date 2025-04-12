import os
# Enable expandable segments to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from transformers import BitsAndBytesConfig

# Assuming DepthAnythingV2 is available; replace with your actual import if needed.
from utils.depth_anything_v2.dpt import DepthAnythingV2

### Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo LoRA test script with depth control support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrained_model", type=str, default="./models",
                        help="Path to pretrained model base directory")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA file to test")
    parser.add_argument("--alpha", type=int, default=128,
                        help="LoRA alpha, defaults to 128")
    parser.add_argument("--output_dir", type=str, default="./test/test_lora",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for inference")
    parser.add_argument("--width", type=int, default=512,
                        help="Width for inference")
    parser.add_argument("--height", type=int, default=512,
                        help="Height for inference")
    parser.add_argument("--num_frames", type=int, default=33,
                        help="Number of frames per video, must be divisible by 4+1")
    parser.add_argument("--inference_steps", type=int, default=20,
                        help="Number of steps for inference")
    parser.add_argument("--prompt", type=str, default="A person typing on a laptop keyboard",
                        help="Prompt for inference")
    parser.add_argument("--control_video", type=str, default=None,
                        help="Path to control video for depth control LoRA")
    parser.add_argument("--depth_model_path", type=str,
                        default="./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
                        help="Path to DepthAnythingV2 model checkpoint")
    parser.add_argument("--skip_base_inference",
                        action="store_true",
                        help="Skip the base model inference step")
    args = parser.parse_args()
    return args

### Process Control Video to Generate Control Latents
def process_control_video(video_path, height, width, num_frames, vae, device, depth_model_path):
    # Load depth model
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device)
    depth_model.eval()

    # Load video using decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # shape: [num_frames, H, W, 3]

    # Resize frames using Lanczos resampling
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
        resized_frames.append(np.array(img))
    video_array = np.stack(resized_frames)  # shape: [num_frames, height, width, 3]

    # Compute depth maps for each frame
    depth_maps = []
    for frame in video_array:
        depth = depth_model.infer_image(frame).cpu()
        depth_maps.append(depth)
    del depth_model
    gc.collect()
    torch.cuda.empty_cache()

    depth_array = np.stack(depth_maps)  # shape: [num_frames, height, width]

    # Normalize depth maps to [-1, 1]
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    if depth_max > depth_min:
        depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 2 - 1
    else:
        depth_array = np.zeros_like(depth_array) - 1

    # Convert single-channel depth to 3 channels for VAE encoding
    depth_array = np.stack([depth_array] * 3, axis=-1)  # shape: [num_frames, height, width, 3]

    # Create tensor and permute dimensions to [B, C, F, H, W]
    depth_tensor = torch.from_numpy(depth_array).permute(3, 0, 1, 2).to(torch.bfloat16).to(device)
    depth_tensor = depth_tensor.unsqueeze(0)
    
    # Encode to latents using VAE and scale by VAE configuration
    with torch.no_grad():
        latents = vae.encode(depth_tensor).latent_dist.sample() * vae.config.scaling_factor

    del depth_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return latents  # expected shape: [1, control_channels, F, H', W']

### Main Inference Function
@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # (Optional) Base Inference
    if not args.skip_base_inference:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model,
            transformer=transformer,
            torch_dtype=torch.float16
        )
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
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        ).frames[0]
        export_to_video(output_base, os.path.join(args.output_dir, "output_base.mp4"), fps=15)
        print("Base model inference completed.")
        print("After base inference:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        del output_base, transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()

        print("After deleting base pipeline:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("Skipping base model inference as requested.")

    # LoRA Inference
    if args.lora:
        print("Running LoRA inference...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16
        )
        print("After loading quantized transformer for LoRA:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        control_latents = None

        # Process control video and update projection BEFORE loading LoRA weights
        if args.control_video:
            from diffusers import AutoencoderKLHunyuanVideo
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                args.pretrained_model,
                subfolder="vae",
                torch_dtype=torch.bfloat16
            ).to(device)
            # Obtain control latents; expected shape: [1, control_channels, F, H, W]
            control_latents = process_control_video(
                args.control_video,
                args.height,
                args.width,
                args.num_frames,
                vae,
                device,
                args.depth_model_path
            )
            del vae
            gc.collect()
            torch.cuda.empty_cache()

            # Update the projection layer.
            # Search for any key starting with "x_embedder.proj.lora_A" in the LoRA checkpoint.
            lora_sd = load_file(args.lora)
            proj_key = None
            for key in lora_sd.keys():
                if key.startswith("x_embedder.proj.lora_A"):
                    proj_key = key
                    break
            
            original_in_channels = transformer.config.in_channels # Store original value (e.g., 16)

            if proj_key is not None:
                proj = transformer.x_embedder.proj
                if hasattr(proj, "base_layer"):
                    base_proj = proj.base_layer
                else:
                    base_proj = proj
                
                # IMPORTANT: Use the stored original_in_channels
                old_in_dim = original_in_channels 
                new_in_dim = old_in_dim + control_latents.shape[1]  # e.g. 16 + 16 = 32
                
                # Sanity-check: the LoRA checkpoint should expect new_in_dim channels.
                assert lora_sd[proj_key].shape[1] == new_in_dim, (
                    f"LoRA checkpoint expects {lora_sd[proj_key].shape[1]} input channels, "
                    f"but new_in_dim is {new_in_dim}"
                )
                in_cls = proj.__class__
                new_proj = in_cls(
                    new_in_dim,
                    base_proj.out_channels,
                    kernel_size=base_proj.kernel_size,
                    stride=base_proj.stride,
                    padding=base_proj.padding,
                ).to(device, torch.bfloat16)
                new_proj.weight.zero_()
                new_proj.bias.zero_()
                # Copy original weights (for the first old_in_dim channels)
                new_proj.weight.data[:, :old_in_dim].copy_(base_proj.weight.data)
                new_proj.bias.data.copy_(base_proj.bias.data)
                if hasattr(proj, "base_layer"):
                    proj.base_layer = new_proj
                else:
                    transformer.x_embedder.proj = new_proj
                print(f"Projection layer updated to accept {new_in_dim} channels.")
            else:
                print("LoRA checkpoint does not contain expected projection key; skipping projection update.")
                new_in_dim = original_in_channels # Should still be 16 here


            print("After processing control video and potentially updating projection:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            lora_sd = load_file(args.lora)
            # If no control video, ensure this is set reasonably
            original_in_channels = transformer.config.in_channels


        # Load the LoRA adapter
        # Make sure LoRA is loaded *after* the projection layer is potentially modified
        # (Loading LoRA might overwrite layers if keys match, though unlikely for base_layer change)
        rank = next(iter(lora_sd.values())).shape[0]
        alpha = args.alpha
        lora_weight = alpha / rank
        print(f"LoRA rank={rank}, alpha={alpha}, lora_weight={lora_weight}")
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        transformer.set_adapters(adapter_names="default_lora", weights=lora_weight)

        # Monkey-patch the forward method to handle control latents
        if control_latents is not None:
            # Attach control latents and original channel count
            transformer.control_latents = control_latents  # shape: [B, C_ctrl, F, H', W'] (B=1 typically)
            transformer.original_in_channels = original_in_channels # e.g., 16
            orig_forward = transformer.forward

            def new_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance, attention_kwargs=None, return_dict=False):
                # hidden_states shape: [Batch*CFG, C, F, H', W'] (Should always be C=16 now)
                current_channels = hidden_states.shape[1]
                
                # Retrieve dimensions stored on the transformer
                original_channels = transformer.original_in_channels
                local_control_latents = transformer.control_latents # shape [1, C_ctrl, F, H', W']

                # Assert that input has original channels, as config wasn't changed
                assert current_channels == original_channels, \
                    f"Input hidden_states have {current_channels} channels, but expected {original_channels}."

                control_channels = local_control_latents.shape[1]
                
                # Handle CFG batching (e.g., repeat control_latents from B=1 to B=2)
                num_batches = hidden_states.shape[0] # e.g., 2 for CFG
                control_batch_size = local_control_latents.shape[0] # e.g., 1
                if num_batches % control_batch_size != 0:
                     raise ValueError("hidden_states batch size must be divisible by control_latents batch size")
                repeat_factor = num_batches // control_batch_size
                
                # Repeat control latents and ensure correct device/dtype
                control_latents_batch = local_control_latents.repeat(repeat_factor, 1, 1, 1, 1)
                control_latents_batch = control_latents_batch.to(hidden_states.device, dtype=hidden_states.dtype)

                # Concatenate along the channel dimension to create the input for the modified layer
                processed_hidden_states = torch.cat([hidden_states, control_latents_batch], dim=1)
                expected_new_channels = original_channels + control_channels
                assert processed_hidden_states.shape[1] == expected_new_channels, \
                     f"Concatenation failed: resulted in {processed_hidden_states.shape[1]} channels, expected {expected_new_channels}"

                # Call the original forward method with the combined input (e.g., 32 channels)
                # The output (model_output) should have original_channels (e.g., 16)
                model_output = orig_forward(
                    processed_hidden_states, 
                    timestep, 
                    encoder_hidden_states, 
                    encoder_attention_mask, 
                    pooled_projections, 
                    guidance, 
                    attention_kwargs=attention_kwargs, 
                    return_dict=return_dict
                )

                # If return_dict is True, model_output is a tuple/dict, handle appropriately
                # Assuming the noise prediction is the first element if not a dict
                if return_dict:
                     # Adapt based on actual structure if needed (HunyuanVideoTransformer3DOutput uses .sample)
                     output_obj = model_output # Assume it's the expected output object
                     pred_noise = output_obj.sample 
                     assert pred_noise.shape[1] == original_channels, \
                         f"Model output (dict) has {pred_noise.shape[1]} channels, expected {original_channels}."
                     return output_obj # Return the full dict/object
                else:
                    # Even with return_dict=False, it seems to return a tuple. Assume noise is the first element.
                    pred_noise = model_output[0] 
                    assert pred_noise.shape[1] == original_channels, \
                        f"Model output (tuple) has {pred_noise.shape[1]} channels, expected {original_channels}."
                    return pred_noise # Return only the noise tensor

            transformer.forward = new_forward
        # End of Monkey-patch

        # Rebuild the pipeline using the modified transformer
        # The pipeline will now use the *original* config.in_channels (16) for latents
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model,
            transformer=transformer,
            torch_dtype=torch.float16
        )

        # Apply more aggressive VAE tiling BEFORE enabling offload for the control lora inference
        print("Applying aggressive VAE tiling settings...")
        pipe.vae.enable_tiling(
            tile_sample_min_height=128,        # Lowered from 256
            tile_sample_min_width=128,         # Lowered from 256
            tile_sample_min_num_frames=16,     # Lowered from 64 (must be <= num_frames)
            tile_sample_stride_height=64,      # Lowered from 192 (more overlap)
            tile_sample_stride_width=64,       # Lowered from 192 (more overlap)
            tile_sample_stride_num_frames=8,   # Lowered from 16 (more temporal overlap)
        )
        
        pipe.enable_sequential_cpu_offload()

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
