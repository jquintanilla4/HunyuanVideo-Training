import os
import gc
import argparse
import torch
import bitsandbytes as bnb
import numpy as np
from PIL import Image
import decord
import imageio
from torchvision.transforms import v2, InterpolationMode
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from peft import PeftModel, set_peft_model_state_dict

from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
from safetensors.torch import load_file

from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
from diffusers.utils import export_to_video, logging
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks

try:
    from utils.depth_anything_v2.dpt import DepthAnythingV2
    print("Imported DepthAnythingV2")
except ImportError:
    print("WARNING: Could not import DepthAnythingV2. Depth preprocessing will fail.")
    DepthAnythingV2 = None  # Placeholder

logger = logging.get_logger(__name__)

# --- Custom Control LoRA Pipeline ---
class HunyuanControlLoraPipeline(HunyuanVideoPipeline):
    DEFAULT_PROMPT_TEMPLATE = {
        "template": "{}",
        "positive": "{text}",
        "negative": "",
        "text_embedding": "{text}",
    }

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], "PipelineCallback", "MultiPipelineCallbacks"]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = None,
        max_sequence_length: int = 256,
        # --- Block Swap Args ---
        double_blocks_to_swap: int = 0,  # Default to 0 (no swap)
        single_blocks_to_swap: int = 0,  # Default to 0 (no swap)
    ):
        prompt_template = prompt_template if prompt_template is not None else self.DEFAULT_PROMPT_TEMPLATE

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )
        if control_latents is not None:
            if self.transformer.config.in_channels % 2 != 0:
                logger.warning(
                    f"Control latents provided, but transformer input channels ({self.transformer.config.in_channels}) is not even. This might indicate the transformer wasn't modified correctly."
                )
            if latents is not None and latents.shape[1] * 2 != self.transformer.config.in_channels:
                logger.warning(
                    f"Provided `latents` shape {latents.shape} channel dim {latents.shape[1]} does not match expected original channels ({self.transformer.config.in_channels // 2}) for control."
                )
            if control_latents.shape[2:] != (
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ):
                logger.warning(
                    f"Provided `control_latents` shape {control_latents.shape} spatial/temporal dims don't match expected latent dims based on height/width/num_frames."
                )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        main_device = self._execution_device  # Use the pipeline's execution device
        offload_device = torch.device("cpu")  # Offload to CPU

        # Validate swap counts
        max_double_blocks = len(getattr(self.transformer, "double_blocks", []))
        max_single_blocks = len(getattr(self.transformer, "single_blocks", []))
        double_blocks_to_swap = min(double_blocks_to_swap, max_double_blocks)
        single_blocks_to_swap = min(single_blocks_to_swap, max_single_blocks)

        if double_blocks_to_swap > 0 or single_blocks_to_swap > 0:
            logger.info(
                f"Block Swapping Enabled: Swapping {double_blocks_to_swap}/{max_double_blocks} double blocks and {single_blocks_to_swap}/{max_single_blocks} single blocks."
            )
            # --- Initial Offload ---
            try:
                logger.info(f"Initial offload to {offload_device}...")
                if hasattr(self.transformer, "double_blocks"):
                    for i in range(double_blocks_to_swap):
                        self.transformer.double_blocks[i].to(offload_device)
                if hasattr(self.transformer, "single_blocks"):
                    for i in range(single_blocks_to_swap):
                        self.transformer.single_blocks[i].to(offload_device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Initial offload complete.")
            except Exception as e:
                logger.error(f"Error during initial block offload: {e}")
                # Continue without swapping if initial offload fails
                double_blocks_to_swap = 0
                single_blocks_to_swap = 0
        else:
            logger.info("Block Swapping Disabled.")

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
            device=main_device,  # Encode prompts on main device
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            main_device,  # Timesteps on main device
            sigmas=sigmas,
        )

        if control_latents is not None:
            num_channels_latents = self.transformer.config.in_channels // 2
            if self.transformer.config.in_channels % 2 != 0:
                logger.warning("Transformer input channels not even, assuming base channel count is same as input.")
                num_channels_latents = self.transformer.config.in_channels
            if control_latents.shape[1] != num_channels_latents:
                raise ValueError(
                    f"Control latents channel dim ({control_latents.shape[1]}) must match base model channel dim ({num_channels_latents})"
                )
        else:
            num_channels_latents = self.transformer.config.in_channels

        # We calculate num_latent_frames here for validation/logging,
        # but prepare_latents needs the original num_frames.
        num_latent_frames_expected = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        if latents is None:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,  # <--- Pass the original num_frames here
                torch.float32,
                main_device,  # Latents on main device
                generator,
                latents=None,
            )
            # Optional: Verify the shape after prepare_latents
            if latents.shape[2] != num_latent_frames_expected:
                logger.warning(
                    f"Prepared latents have {latents.shape[2]} frames, but expected {num_latent_frames_expected} based on num_frames and temporal scale factor."
                )

        else:
            if control_latents is not None and latents.shape[1] != num_channels_latents:
                raise ValueError(
                    f"Provided `latents` have {latents.shape[1]} channels, but expected {num_channels_latents} for control model base."
                )
            elif control_latents is None and latents.shape[1] != num_channels_latents:
                raise ValueError(
                    f"Provided `latents` have {latents.shape[1]} channels, but expected {num_channels_latents} for standard model."
                )
            latents = latents.to(device=main_device, dtype=torch.float32)

        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=main_device) * 1000.0

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # --- Block Swapping: Move to Main Device ---
                if double_blocks_to_swap > 0 or single_blocks_to_swap > 0:
                    try:
                        if hasattr(self.transformer, "double_blocks"):
                            for j in range(double_blocks_to_swap):
                                self.transformer.double_blocks[j].to(main_device)
                        if hasattr(self.transformer, "single_blocks"):
                            for j in range(single_blocks_to_swap):
                                self.transformer.single_blocks[j].to(main_device)
                    except Exception as e:
                        logger.error(f"Error moving blocks to main device at step {i}: {e}")

                # Prepare model input
                if control_latents is not None:
                    control_latents_input = control_latents.to(device=main_device, dtype=transformer_dtype)
                    model_input = torch.cat([latents.to(transformer_dtype), control_latents_input], dim=1)
                else:
                    model_input = latents.to(transformer_dtype)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                timestep = timestep.to(transformer_dtype)

                # Predict noise
                noise_pred = self.transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # --- Block Swapping: Move back to Offload Device ---
                if double_blocks_to_swap > 0 or single_blocks_to_swap > 0:
                    try:
                        if hasattr(self.transformer, "double_blocks"):
                            for j in range(double_blocks_to_swap):
                                self.transformer.double_blocks[j].to(offload_device)
                        if hasattr(self.transformer, "single_blocks"):
                            for j in range(single_blocks_to_swap):
                                self.transformer.single_blocks[j].to(offload_device)
                        # Optional: Force memory cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error moving blocks to offload device at step {i}: {e}")

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # --- Final Offload (if swapping was enabled) ---
        if double_blocks_to_swap > 0 or single_blocks_to_swap > 0:
            try:
                logger.info(f"Final offload to {offload_device}...")
                if hasattr(self.transformer, "double_blocks"):
                    for i in range(double_blocks_to_swap):
                        self.transformer.double_blocks[i].to(offload_device)
                if hasattr(self.transformer, "single_blocks"):
                    for i in range(single_blocks_to_swap):
                        self.transformer.single_blocks[i].to(offload_device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Final offload complete.")
            except Exception as e:
                logger.error(f"Error during final block offload: {e}")

        # Decode latents
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)

# --- Helper Functions (from training script) ---
@contextmanager
def timer(message=""):
    from time import perf_counter

    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"{message} {end_time - start_time:0.2f} seconds")

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo lora/control-lora test script",
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
    # --- Control LoRA Args ---
    parser.add_argument(
        "--control_lora",
        action="store_true",
        help="Load LoRA as control lora (requires modified input layer and control_input)",
    )
    parser.add_argument(
        "--control_lora_weight",
        type=float,
        default=1.0,
        help="Weight to apply to the control LoRA adapter (only used with --control_lora)",
    )
    parser.add_argument(
        "--control_input",
        type=str,
        default=None,
        help="Path to control image or video file (required for control_lora)",
    )
    parser.add_argument(
        "--control_preprocess",
        type=str,
        default="depth",
        choices=["depth"],  # Add more if needed
        help="Preprocess to apply to control_input",
    )
    parser.add_argument(
        "--depth_model_path",
        type=str,
        default="./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
        help="Path to Depth Anything V2 model weights",
    )
    # --- Standard Args ---
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,  # Will default to rank if None
        help="lora alpha, defaults to rank",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test/test_control",
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
        default=49,  # Must be (n * 4) + 1; was initially 33
        help="Number of frames per video, must be divisible by 4+1",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=30,
        help="Number of steps for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person typing on a laptop keyboard",
        help="Prompt for inference",
    )
    # --- Block Swap Args ---
    parser.add_argument(
        "--double_blocks_swap",
        type=int,
        default=0,
        help="Number of double transformer blocks to offload to CPU (0 = none).",
    )
    parser.add_argument(
        "--single_blocks_swap",
        type=int,
        default=0,
        help="Number of single transformer blocks to offload to CPU (0 = none).",
    )
    # ---------------------

    args = parser.parse_args()

    if args.control_lora and args.control_input is None:
        parser.error("--control_input is required when using --control_lora")
    if args.control_lora and args.lora is None:
        parser.error("--lora checkpoint path is required when using --control_lora")
    if args.control_lora and args.control_preprocess == "depth" and DepthAnythingV2 is None:
        parser.error("DepthAnythingV2 model could not be imported, cannot use --control_lora with depth preprocessing.")

    return args

# --- Control Preprocessing ---
def preprocess_control(args, pixels, depth_model, device):
    """
    Generates control tensor (e.g., depth) from input pixels.
    pixels: Input tensor (B, C, F, H, W), range [-1, 1]
    Returns: Control tensor (B, 3, F, H, W), range [-1, 1], dtype=torch.float16
    """
    if args.control_preprocess == "depth":
        if depth_model is None:
            raise ValueError("Depth model not loaded, cannot preprocess for depth.")

        B, C, F, H, W = pixels.shape
        depth_tensor = torch.zeros((B, F, H, W), device=pixels.device, dtype=torch.float32)  # Use float32 for processing

        # --- List to store depth frames for video ---
        depth_frames_for_video = []

        print(f"Preprocessing {B}x{F} frames for depth...")
        for b in range(B):
            if b > 0:
                depth_frames_for_video = []

            for f in range(F):
                # Convert frame to format expected by Depth Anything V2 (H, W, C), range [0, 1]
                frame = pixels[b, :, f].float() * 0.5 + 0.5  # Denormalize [-1,1] -> [0,1]
                frame_np = frame.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                depth = depth_model.infer_image(frame_np)  # Returns numpy H W

                # Resize depth map back to original H, W if needed (model might output different size)
                depth_h, depth_w = depth.shape
                if depth_h != H or depth_w != W:
                    # Note: resizing might put tensor on 'device' (GPU)
                    depth_tensor_resized = torch.tensor(depth, device=device).unsqueeze(0).unsqueeze(0)  # 1, 1, dH, dW
                    depth_tensor_resized = v2.functional.resize(
                        depth_tensor_resized,
                        size=[H, W],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    # --- Explicitly move resized tensor to CPU before NumPy ---
                    depth = depth_tensor_resized.squeeze().cpu().numpy()
                # else: depth is already a NumPy array from infer_image

                # Normalize depth map for visualization (0 to 1) - ensure it's numpy
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_normalized_01 = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized_01 = np.zeros_like(depth)  # depth is numpy here

                # --- Store frame for video ---
                try:
                    # --- Force conversion to numpy right before use ---
                    if isinstance(depth_normalized_01, torch.Tensor):
                        print(f"WARN: Converting depth_normalized_01 from Tensor to ndarray at frame {f}")
                        current_depth_np = depth_normalized_01.cpu().numpy()
                    elif isinstance(depth_normalized_01, np.ndarray):
                        current_depth_np = depth_normalized_01  # Already numpy
                    else:
                        # Attempt conversion if it's some other type
                        print(
                            f"WARN: Unexpected type for depth_normalized_01 ({type(depth_normalized_01)}), attempting np.array()"
                        )
                        current_depth_np = np.array(depth_normalized_01)

                    depth_img_np = (current_depth_np * 255).astype(np.uint8)
                    # --- END FORCE CONVERSION ---
                    depth_frames_for_video.append(depth_img_np)
                except Exception as e:
                    # Add type info to error
                    print(f"WARN: Failed to store depth frame {b}-{f} for video (type: {type(depth_normalized_01)}): {e}")

                # Scale depth map to [-1, 1] for VAE input
                # depth_normalized_01 is numpy from normalization step
                depth_scaled_np = depth_normalized_01 * 2.0 - 1.0
                depth_tensor[b, f] = torch.tensor(depth_scaled_np, device=device, dtype=torch.float32)

            # --- Save depth video for this batch item ---
            if depth_frames_for_video:
                output_video_path = os.path.join(args.output_dir, f"debug_depth_video_batch_{b:02d}.mp4")
                print(f"DEBUG: Saving depth video ({len(depth_frames_for_video)} frames) to: {output_video_path}")
                try:
                    rgb_depth_frames = np.stack([depth_frames_for_video] * 3, axis=-1)
                    imageio.mimsave(output_video_path, rgb_depth_frames, fps=24)
                except Exception as e:
                    print(f"ERROR: Failed to save debug depth video: {e}")

        # Reshape and replicate to 3 channels for VAE
        depth_tensor = depth_tensor.unsqueeze(1)  # (B, 1, F, H, W)
        control = depth_tensor.repeat(1, 3, 1, 1, 1).to(dtype=torch.float16)  # (B, 3, F, H, W)

        # Assertions
        assert control.shape == (B, 3, F, H, W), f"Unexpected control shape: {control.shape}"
        assert not torch.isnan(control).any(), "NaN values detected in control tensor"
        # Relaxed check due to potential float precision issues near -1/1
        assert control.min() >= -1.01 and control.max() <= 1.01, (
            f"Control values out of ~[-1,1] range: min={control.min().item()}, max={control.max().item()}"
        )

        print("Depth preprocessing complete.")
        return control
    else:
        raise NotImplementedError(f"Control preprocessing '{args.control_preprocess}' not implemented.")

# --- Main Inference ---
@torch.inference_mode()
def main(args):
    decord.bridge.set_bridge("torch")
    device = torch.cuda.current_device()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)  # Use CPU generator for reproducibility

    # --- Load Models (VAE, Depth, Text Encoders) ---
    with timer("Loading VAE"):
        vae = AutoencoderKLHunyuanVideo.from_pretrained(args.pretrained_model, subfolder="vae").to(
            device=device, dtype=torch.float16
        )
        vae.requires_grad_(False)
        vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_min_num_frames=max(64, args.num_frames * 2),  # Adjust based on num_frames
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
            tile_sample_stride_num_frames=max(16, args.num_frames // 2),
        )

    depth_model = None
    if args.control_lora and args.control_preprocess == "depth":
        with timer("Loading Depth Model"):
            if not os.path.exists(args.depth_model_path):
                raise FileNotFoundError(f"Depth model not found at {args.depth_model_path}. Please download or provide correct path.")
            if DepthAnythingV2 is None:
                raise ImportError("DepthAnythingV2 could not be imported.")
            depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            depth_model.load_state_dict(torch.load(args.depth_model_path, map_location="cpu", weights_only=True))
            depth_model = depth_model.to(device)
            depth_model.requires_grad_(False)
            depth_model.eval()

    with timer("Loading Text Encoders/Tokenizers"):
        tokenizer_clip = CLIPTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
        text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2").to(
            device=device, dtype=torch.bfloat16
        )
        tokenizer_llama = LlamaTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        text_encoder_llama = LlamaModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(
            device=device, dtype=torch.bfloat16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    with timer("Loading Transformer"):
        # Load initially to main device for modifications/LoRA application
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)

    if args.control_lora:
        print("Modifying transformer input layer for Control LoRA...")
        with torch.no_grad():
            original_in_channels = transformer.config.in_channels
            in_proj_layer = transformer.x_embedder.proj
            in_cls = in_proj_layer.__class__
            old_in_dim = original_in_channels
            new_in_dim = old_in_dim * 2

            print(f"  Original input channels: {old_in_dim}")
            print(f"  New input channels: {new_in_dim}")

            new_in = in_cls(
                new_in_dim,
                in_proj_layer.out_channels,
                in_proj_layer.kernel_size,
                in_proj_layer.stride,
                in_proj_layer.padding,
            ).to(device=in_proj_layer.weight.device, dtype=in_proj_layer.weight.dtype)

            new_in.weight.zero_()
            new_in.bias.zero_()
            new_in.weight[:, :old_in_dim].copy_(in_proj_layer.weight)
            new_in.bias.copy_(in_proj_layer.bias)

            transformer.x_embedder.proj = new_in
            transformer.register_to_config(in_channels=new_in_dim)
            print("  Input layer modified and config updated.")
        gc.collect()
        torch.cuda.empty_cache()

    if args.lora is not None:
        print(f"Loading LoRA adapter from: {args.lora}")
        lora_sd = load_file(args.lora, device="cpu")  # Load to CPU first

        if args.control_lora:
            rank = 0  # Infer rank from loaded state dict
            for key in lora_sd.keys():
                if ".lora_A.weight" in key:
                    rank = lora_sd[key].shape[0]
                    break
            if rank == 0:
                print("WARNING: Could not infer LoRA rank from state dict keys.")
                rank = args.alpha if args.alpha is not None else 128  # Default fallback rank
                print(f"Falling back to rank: {rank}")

            alpha = args.alpha if args.alpha is not None else rank  # Default alpha to rank if not given
            lora_weight = args.control_lora_weight
            print(f"Control LoRA: Inferred rank={rank}, alpha={alpha}, Using specified weight={lora_weight}")
        else:
            rank = 0
            for key in lora_sd.keys():
                if ".lora_A.weight" in key:
                    rank = lora_sd[key].shape[0]
                    break
            if rank == 0:
                print("WARNING: Could not infer LoRA rank from state dict keys.")
                rank = args.alpha if args.alpha is not None else 128  # Default fallback rank
                print(f"Falling back to rank: {rank}")

            alpha = args.alpha if args.alpha is not None else rank
            lora_weight = alpha / rank if rank > 0 else 1.0  # Avoid division by zero
            print(f"Standard LoRA: Inferred rank={rank}, alpha={alpha}, weight={lora_weight:.4f}")

        # Load the LoRA adapter using the state dictionary
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")

        # Set the adapter with the computed weight
        transformer.set_adapters(adapter_names=["default_lora"], weights=[lora_weight])
        print(f"Set adapter 'default_lora' with weight {lora_weight}")

        del lora_sd
        gc.collect()
        torch.cuda.empty_cache()

    transformer = transformer.to(device)

    control_latents = None
    if args.control_lora:
        with timer(f"Loading and preprocessing control input '{args.control_input}'"):
            ext = os.path.splitext(args.control_input)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                image = Image.open(args.control_input).convert("RGB")
                target_height, target_width = args.height, args.width
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                pixels = v2.functional.to_tensor(image)  # C, H, W, range [0, 1]
                pixels = pixels * 2.0 - 1.0  # Scale to [-1, 1]
                pixels = pixels.unsqueeze(0).repeat(1, 1, args.num_frames, 1, 1)  # B, C, F, H, W (B=1)
            elif ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                vr = decord.VideoReader(args.control_input, height=args.height, width=args.width)
                indices = np.linspace(0, len(vr) - 1, args.num_frames, dtype=int)
                frames = vr.get_batch(indices).float()  # F, H, W, C, range [0, 255]
                pixels = frames.permute(3, 0, 1, 2) / 255.0  # C, F, H, W, range [0, 1]
                pixels = pixels * 2.0 - 1.0  # Scale to [-1, 1]
                pixels = pixels.unsqueeze(0)  # B, C, F, H, W (B=1)
            else:
                raise ValueError(f"Unsupported control input file type: {ext}")

            pixels = pixels.to(device=device, dtype=torch.float16)

            control_pixels = preprocess_control(args, pixels, depth_model, device)

            vae.enable_tiling()
            control_latents = vae.encode(control_pixels).latent_dist.sample() * vae.config.scaling_factor
            control_latents = control_latents.to(transformer.dtype)

            print(f"Control latents generated with shape: {control_latents.shape}")
            del pixels, control_pixels
            gc.collect()
            torch.cuda.empty_cache()

    pipe = HunyuanControlLoraPipeline(
        vae=vae,
        text_encoder=text_encoder_llama,
        tokenizer=tokenizer_llama,
        transformer=transformer,
        scheduler=scheduler,
        text_encoder_2=text_encoder_clip,
        tokenizer_2=tokenizer_clip,
    )
    pipe = pipe.to(device)
    pipe.vae.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_num_frames=max(64, args.num_frames * 2),
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
        tile_sample_stride_num_frames=max(16, args.num_frames // 2),
    )

    output_filename = "output_base.mp4" if args.lora is None else "output_lora.mp4"
    if args.control_lora:
        output_filename = "output_control_lora.mp4"

    print(f"Running inference for prompt: '{args.prompt}'")
    print(f"Outputting to: {os.path.join(args.output_dir, output_filename)}")

    output = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        generator=generator,
        control_latents=control_latents,
        prompt_template=pipe.DEFAULT_PROMPT_TEMPLATE,
        # --- Pass Block Swap Args ---
        double_blocks_to_swap=args.double_blocks_swap,
        single_blocks_to_swap=args.single_blocks_swap,
    ).frames[0]

    export_to_video(
        output,
        os.path.join(args.output_dir, output_filename),
        fps=24,
    )
    print("Inference complete.")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)