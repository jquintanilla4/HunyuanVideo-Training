import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import argparse
import torch
import numpy as np
import torch.nn as nn
import math
from safetensors.torch import load_file
from PIL import Image
import decord

from diffusers import (
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
    AutoencoderKLHunyuanVideo,
)
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig

# Import any required functions from your local utils (e.g., retrieve_timesteps, randn_tensor, etc.)
# from your_utils import retrieve_timesteps, randn_tensor  # adjust as necessary

# Assuming DepthAnythingV2 is available; replace with your actual import if needed.
from utils.depth_anything_v2.dpt import DepthAnythingV2

# DEFAULT_PROMPT_TEMPLATE should be defined or imported as appropriate.
DEFAULT_PROMPT_TEMPLATE = {"template": "{}"}

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
    return parser.parse_args()

def process_control_video(video_path, height, width, num_frames, vae, device, depth_model_path):
    # Load and prepare the DepthAnythingV2 model.
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device)
    depth_model.eval()

    # Load video frames using decord.
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # shape: [num_frames, H, W, 3]

    # Resize frames.
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
        resized_frames.append(np.array(img))
    video_array = np.stack(resized_frames)

    # Compute depth maps.
    depth_maps = []
    for frame in video_array:
        depth = depth_model.infer_image(frame).cpu()
        depth_maps.append(depth)
    del depth_model
    gc.collect()
    torch.cuda.empty_cache()

    depth_array = np.stack(depth_maps)
    # Normalize to [-1, 1]
    dmin, dmax = depth_array.min(), depth_array.max()
    depth_array = (depth_array - dmin) / (dmax - dmin) * 2 - 1 if dmax > dmin else np.zeros_like(depth_array) - 1
    # Expand single channel to 3 channels.
    depth_array = np.stack([depth_array] * 3, axis=-1)
    # Convert to tensor [B, C, F, H, W]
    depth_tensor = torch.from_numpy(depth_array).permute(3, 0, 1, 2).to(torch.bfloat16).to(device)
    depth_tensor = depth_tensor.unsqueeze(0)
    with torch.no_grad():
        latents = vae.encode(depth_tensor).latent_dist.sample() * vae.config.scaling_factor
    del depth_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return latents

# --- Custom __call__ method for the pipeline (patching the denoising loop) ---
def custom_call(
    self,
    prompt=None,
    prompt_2=None,
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=50,
    sigmas=None,
    guidance_scale=6.0,
    num_videos_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    prompt_attention_mask=None,
    output_type="pil",
    return_dict=True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    max_sequence_length=256,
):
    # 1. Check inputs, encode prompt, etc.
    self.check_inputs(prompt, prompt_2, height, width, prompt_embeds, callback_on_step_end_tensor_inputs, prompt_template)
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._interrupt = False
    device = self._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_template=prompt_template,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

    # 5. Prepare latents (Note: latent channel count is still 16 initially)
    num_channels_latents = self.transformer.config.in_channels  # originally 16
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_latent_frames,
        torch.float32,
        device,
        generator,
        latents,
    )
    # 6. Prepare guidance.
    guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

    # 7. Denoising loop with channel splitting.
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    progress_bar = self.progress_bar(total=num_inference_steps)
    progress_bar.__enter__()
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        # Split latents into original and control channels.
        original_channels = (self.transformer.original_in_channels 
                             if hasattr(self.transformer, "original_in_channels") 
                             else self.transformer.config.in_channels)
        latents_orig = latents[:, :original_channels]
        control_latents = latents[:, original_channels:]
        # Apply scheduler step only to the original channels.
        updated_orig = self.scheduler.step(noise_pred, t, latents_orig, return_dict=False)[0]
        latents = torch.cat([updated_orig, control_latents], dim=1)

        if callback_on_step_end is not None:
            callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            progress_bar.update()
    progress_bar.__exit__(None, None, None)

    # 8. Decode only the original channels.
    if output_type != "latent":
        latents_to_decode = latents[:, :original_channels]
        latents_to_decode = latents_to_decode.to(self.vae.dtype) / self.vae.config.scaling_factor
        video = self.vae.decode(latents_to_decode, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    self.maybe_free_model_hooks()
    return video if not return_dict else HunyuanVideoPipelineOutput(frames=video)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Base inference (optional) ---
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
        del output_base, transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()
        print("After deleting base pipeline:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("Skipping base model inference as requested.")

    # --- LoRA inference with control (if provided) ---
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
        lora_sd = load_file(args.lora)
        original_in_channels = transformer.config.in_channels  # typically 16

        # If a control video is provided, process it and update the projection layer.
        if args.control_video:
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                args.pretrained_model,
                subfolder="vae",
                torch_dtype=torch.bfloat16
            ).to(device)
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

            # Update projection layer if the LoRA checkpoint contains the expected key.
            proj_key = None
            for key in lora_sd.keys():
                if key.startswith("x_embedder.proj.lora_A"):
                    proj_key = key
                    break

            if proj_key is not None:
                proj = transformer.x_embedder.proj
                base_proj = proj.base_layer if hasattr(proj, "base_layer") else proj
                old_in_dim = original_in_channels
                new_in_dim = old_in_dim + control_latents.shape[1]  # e.g., 16 + 16 = 32
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
                new_proj.weight.data[:, :old_in_dim].copy_(base_proj.weight.data)
                new_proj.bias.data.copy_(base_proj.bias.data)
                if hasattr(proj, "base_layer"):
                    proj.base_layer = new_proj
                else:
                    transformer.x_embedder.proj = new_proj

                # --- RE-ADD register_to_config after modifying projection ---
                transformer.register_to_config(in_channels=new_in_dim)
                print(f"Projection layer updated to accept {new_in_dim} channels and config registered.")
            else:
                print("LoRA checkpoint does not contain expected projection key; skipping projection update.")
                new_in_dim = original_in_channels
        else:
            lora_sd = load_file(args.lora)
            new_in_dim = original_in_channels

        # Store the original channel count for later reference.
        transformer.original_in_channels = original_in_channels

        # Load the LoRA adapter.
        rank = next(iter(lora_sd.values())).shape[0]
        lora_weight = args.alpha / rank
        print(f"LoRA rank={rank}, alpha={args.alpha}, lora_weight={lora_weight}")
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        transformer.set_adapters(adapter_names="default_lora", weights=lora_weight)

        # Optionally patch transformer.forward to concatenate control latents.
        if control_latents is not None:
            transformer.control_latents = control_latents  # shape: [B, C_control, F, H', W']
            orig_forward = transformer.forward
            def new_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance, attention_kwargs=None, return_dict=False):
                current_channels = hidden_states.shape[1]
                original_channels = transformer.original_in_channels
                local_control_latents = transformer.control_latents
                assert current_channels == original_channels, (
                    f"Expected {original_channels} channels, got {current_channels}"
                )
                control_channels = local_control_latents.shape[1]
                num_batches = hidden_states.shape[0]
                control_batch_size = local_control_latents.shape[0]
                if num_batches % control_batch_size != 0:
                    raise ValueError("hidden_states batch size must be divisible by control_latents batch size")
                repeat_factor = num_batches // control_batch_size
                control_latents_batch = local_control_latents.repeat(repeat_factor, 1, 1, 1, 1)
                control_latents_batch = control_latents_batch.to(hidden_states.device, dtype=hidden_states.dtype)
                processed_hidden_states = torch.cat([hidden_states, control_latents_batch], dim=1)
                expected_new_channels = original_channels + control_channels
                assert processed_hidden_states.shape[1] == expected_new_channels, (
                    f"Concatenation resulted in {processed_hidden_states.shape[1]} channels, expected {expected_new_channels}"
                )
                model_output = orig_forward(
                    processed_hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance, attention_kwargs=attention_kwargs, return_dict=return_dict
                )
                if return_dict:
                    pred_noise = model_output.sample
                    assert pred_noise.shape[1] == original_channels, (
                        f"Model output has {pred_noise.shape[1]} channels, expected {original_channels}"
                    )
                    return model_output
                else:
                    pred_noise = model_output[0]
                    assert pred_noise.shape[1] == original_channels, (
                        f"Model output has {pred_noise.shape[1]} channels, expected {original_channels}"
                    )
                    return pred_noise
            transformer.forward = new_forward

        # Rebuild the pipeline with the modified transformer.
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model,
            transformer=transformer,
            torch_dtype=torch.float16
        )

        # Apply aggressive VAE tiling settings.
        print("Applying aggressive VAE tiling settings...")
        pipe.vae.enable_tiling(
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_min_num_frames=16,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
            tile_sample_stride_num_frames=8,
        )
        pipe.enable_sequential_cpu_offload()

        # Patch the pipeline's __call__ with our custom call function.
        import types
        pipe.__call__ = types.MethodType(custom_call, pipe)

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
