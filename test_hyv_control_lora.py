import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import random
import argparse
import datetime
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision.transforms import v2, InterpolationMode

from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import set_peft_model_state_dict
from PIL import Image

import decord
decord.bridge.set_bridge('torch')

DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


def cache_video(tensor, save_file, fps=24, nrow=1, normalize=True, value_range=(-1, 1)):
    """Save a tensor as a video file."""
    import torchvision.io as io
    import torch.nn.functional as F

    # Prepare tensor for saving
    if normalize:
        tensor = (tensor - value_range[0]) / (value_range[1] - value_range[0])

    tensor = (tensor * 255).to(torch.uint8)

    # Reshape to expected format for make_grid
    b, c, t, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 3, 4, 1)  # B, T, H, W, C

    # Save video
    io.write_video(save_file, tensor[0].cpu(), fps=fps)

    return save_file


@torch.inference_mode()
def main(args):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = os.path.join(args.output_dir, date_time)
    os.makedirs(real_output_dir, exist_ok=True)

    # Load text encoders and tokenizers for text conditioning
    if args.prompt is not None or args.negative_prompt is not None:
        print("Loading text encoders...")

        tokenizer_clip = CLIPTokenizerFast.from_pretrained(
            args.pretrained_model_dir, subfolder="tokenizer_2")
        text_encoder_clip = CLIPTextModel.from_pretrained(
            args.pretrained_model_dir, subfolder="text_encoder_2").to(device=device, dtype=torch.bfloat16)
        text_encoder_clip.requires_grad_(False)

        tokenizer_llama = LlamaTokenizerFast.from_pretrained(
            args.pretrained_model_dir, subfolder="tokenizer")
        text_encoder_llama = LlamaModel.from_pretrained(
            args.pretrained_model_dir, subfolder="text_encoder").to(device=device, dtype=torch.bfloat16)
        text_encoder_llama.requires_grad_(False)

        # Encode prompts
        def encode_clip(prompt):
            input_ids = tokenizer_clip(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(text_encoder_clip.device)

            prompt_embeds = text_encoder_clip(
                input_ids,
                output_hidden_states=False,
            ).pooler_output

            return prompt_embeds

        def encode_llama(
            prompt,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            max_sequence_length=256,
            num_hidden_layers_to_skip=2,
        ):
            prompt = prompt_template["template"].format(prompt)
            crop_start = prompt_template.get("crop_start", None)
            max_sequence_length += crop_start

            text_inputs = tokenizer_llama(
                prompt,
                max_length=max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
            )
            text_input_ids = text_inputs.input_ids.to(
                device=text_encoder_llama.device)
            prompt_attention_mask = text_inputs.attention_mask.to(
                device=text_encoder_llama.device)

            prompt_embeds = text_encoder_llama(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-(num_hidden_layers_to_skip + 1)]

            if crop_start is not None and crop_start > 0:
                prompt_embeds = prompt_embeds[:, crop_start:]
                prompt_attention_mask = prompt_attention_mask[:, crop_start:]

            return prompt_embeds, prompt_attention_mask

        # Process prompt and negative prompt
        pos_clip_embed = encode_clip(args.prompt)
        pos_llama_embed, pos_llama_mask = encode_llama(args.prompt)

        neg_prompt = args.negative_prompt or ""
        neg_clip_embed = encode_clip(neg_prompt)
        neg_llama_embed, neg_llama_mask = encode_llama(neg_prompt)

        # Clean up to save memory
        del tokenizer_clip, text_encoder_clip, tokenizer_llama, text_encoder_llama
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Load from pre-computed embeddings
        print("Loading pre-computed embeddings...")
        embedding_dict = load_file(args.embedding)
        pos_clip_embed = embedding_dict["clip_embed"].to(
            device=device, dtype=torch.bfloat16)
        pos_llama_embed = embedding_dict["llama_embed"].to(
            device=device, dtype=torch.bfloat16)
        pos_llama_mask = embedding_dict["llama_mask"].to(
            device=device, dtype=torch.bfloat16)

        neg_embedding_dict = load_file(args.negative_embedding)
        neg_clip_embed = neg_embedding_dict["clip_embed"].to(
            device=device, dtype=torch.bfloat16)
        neg_llama_embed = neg_embedding_dict["llama_embed"].to(
            device=device, dtype=torch.bfloat16)
        neg_llama_mask = neg_embedding_dict["llama_mask"].to(
            device=device, dtype=torch.bfloat16)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        args.pretrained_model_dir, subfolder="vae").to(device=device, dtype=torch.float16)
    vae.requires_grad_(False)
    vae.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_num_frames=48,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
        tile_sample_stride_num_frames=32,
    )

    # Load diffusion model
    print("Loading diffusion model...")
    if args.quant_type == "nf4":
        quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.quant_type == "int8":
        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    diffusion_model = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_dir,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    diffusion_model.requires_grad_(False)

    # Load LoRA weights and modify input projection if this is a control LoRA
    print(f"Loading LoRA from {args.lora}...")
    lora_sd = load_file(args.lora)

    # Check if this is a control LoRA by looking at in_proj dimensions
    if "in_proj.lora_A.weight" in lora_sd:
        print("Detected control LoRA, modifying input projection layer...")
        # Modify input projection to handle concatenated control input
        with torch.no_grad():
            in_cls = diffusion_model.in_proj.__class__  # nn.Conv3d
            old_in_dim = diffusion_model.config.in_channels  # 32
            new_in_dim = old_in_dim * 2  # Double the channels for concatenated control

            new_in = in_cls(
                new_in_dim,
                diffusion_model.in_proj.out_channels,
                diffusion_model.in_proj.kernel_size,
                diffusion_model.in_proj.stride,
                diffusion_model.in_proj.padding,
            ).to(device=device, dtype=torch.bfloat16)

            new_in.weight.zero_()
            new_in.bias.zero_()

            new_in.weight[:, :old_in_dim].copy_(diffusion_model.in_proj.weight)
            new_in.bias.copy_(diffusion_model.in_proj.bias)

            diffusion_model.in_proj = new_in
            diffusion_model.register_to_config(in_channels=new_in_dim)

    # Load LoRA weights
    set_peft_model_state_dict(diffusion_model, lora_sd)

    gc.collect()
    torch.cuda.empty_cache()

    # Calculate dimensions
    latent_width = args.width // 16
    latent_height = args.height // 16
    latent_frames = (args.frames - 1) // 4 + 1

    frames = (latent_frames - 1) * 4 + 1

    # Set up noise scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(args.steps, device=device)

    # Create initial latents
    torch.manual_seed(args.seed)
    latents = torch.randn(1, 32, latent_frames,
                          latent_height, latent_width).to(device)

    # Process control video if provided
    if args.control_video is not None:
        print(f"Processing control video: {args.control_video}")
        vr = decord.VideoReader(args.control_video)

        # Extract frames and resize to match target
        if len(vr) < frames:
            print(
                f"Warning: Control video has {len(vr)} frames, but {frames} are needed. Will repeat the last frame.")

        # Read frames from control video
        control_frames = min(len(vr), frames)
        control_pixels = vr[:control_frames]

        # If we need more frames, repeat the last frame
        if control_frames < frames:
            last_frame = control_pixels[-1].unsqueeze(0)
            repeat_frames = frames - control_frames
            control_pixels = torch.cat(
                [control_pixels, last_frame.repeat(repeat_frames, 1, 1, 1)], dim=0)

        # Process control video based on specified preprocessing
        if args.control_preprocess == "depth":
            print("Applying depth preprocessing...")
            # Initialize depth model here if needed
            try:
                from utils.depth_anything_v2.dpt import DepthAnythingV2
                depth_model = DepthAnythingV2(
                    encoder='vits', features=64, out_channels=[48, 96, 192, 384])
                depth_model.load_state_dict(torch.load(
                    "./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth", map_location='cpu', weights_only=True))
                depth_model = depth_model.to(device)
                depth_model.requires_grad_(False)
                depth_model.eval()

                # Process each frame to get depth map
                depth_frames = []
                for frame in tqdm(control_pixels, desc="Processing depth maps"):
                    # Convert frame to proper format for depth model
                    frame_rgb = frame.float() / 255.0  # Normalize to 0-1
                    frame_numpy = frame_rgb.permute(
                        1, 2, 0).cpu().numpy()  # HWC

                    # Get depth
                    depth = depth_model.infer_image(frame_numpy)

                    # Normalize depth to -1 to 1
                    depth_min, depth_max = depth.min(), depth.max()
                    if depth_max > depth_min:
                        depth = (depth - depth_min) / (depth_max - depth_min)
                    depth = depth * 2 - 1

                    depth_frames.append(torch.from_numpy(
                        depth).unsqueeze(0))  # Add channel dim

                # Stack frames and repeat channel to match RGB
                control_pixels = torch.stack(depth_frames, dim=0)  # FCHW
                control_pixels = control_pixels.repeat(
                    1, 3, 1, 1)  # Repeat to 3 channels

            except ImportError:
                print("Warning: Depth model not found, falling back to plain video")

        # Reshape and prepare control video
        control_pixels = control_pixels.permute(
            3, 0, 1, 2).unsqueeze(0)  # FHWC -> CFHW -> BCFHW

        # Apply transforms to control video
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(args.height, args.width),
                      interpolation=InterpolationMode.BICUBIC),
        ])

        control_pixels = transform(control_pixels) * 2 - 1
        control_pixels = torch.clamp(
            torch.nan_to_num(control_pixels), min=-1, max=1)

        # Encode control video to latent space
        control_latents = vae.encode(control_pixels.to(
            device=device, dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor

        # Verify shapes
        print(f"Latents shape: {latents.shape}")
        print(f"Control latents shape: {control_latents.shape}")

        assert control_latents.shape[2:] == latents.shape[2:
                                                          ], f"Latent shapes don't match: {control_latents.shape} vs {latents.shape}"

    # Set up guidance scale
    guidance_scale = torch.tensor(
        [args.cfg] * latents.shape[0], dtype=torch.float32, device=device) * 1000.0

    # Run diffusion process
    print("Running diffusion process...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for t in tqdm(scheduler.timesteps):
            # Format inputs
            timesteps = torch.full((1,), t, device=device, dtype=torch.long)

            # Create noisy input
            sigma = t / 1000.0
            noisy_model_input = latents

            # Concatenate control latents if using control LoRA
            if args.control_video is not None:
                noisy_model_input = torch.cat(
                    [noisy_model_input, control_latents], dim=1)

            # Run conditional prediction
            cond_output = diffusion_model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=pos_llama_embed,
                encoder_attention_mask=pos_llama_mask,
                pooled_projections=pos_clip_embed,
                guidance=guidance_scale,
                return_dict=False,
            )[0]

            # Run unconditional prediction for classifier-free guidance
            uncond_output = diffusion_model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=neg_llama_embed,
                encoder_attention_mask=neg_llama_mask,
                pooled_projections=neg_clip_embed,
                guidance=guidance_scale,
                return_dict=False,
            )[0]

            # Apply classifier-free guidance
            noise_pred = uncond_output + args.cfg * \
                (cond_output - uncond_output)

            # Update latents
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode final latents
        print("Decoding final video...")
        decoded_video = vae.decode(latents).sample.to(device)

        # Save control video side by side with generated video if control was used
        if args.control_video is not None:
            # Resize control for display
            control_display = control_pixels.to(device)

            # Concatenate videos side by side
            if args.width > args.height:
                cat_dim = -2  # Stack vertically
            else:
                cat_dim = -1  # Stack horizontally

            combined_video = torch.cat(
                [control_display, decoded_video], dim=cat_dim)
            output_path = os.path.join(
                real_output_dir, "control_and_output.mp4")

            # Save the video
            cache_video(
                tensor=combined_video,
                save_file=output_path,
                fps=16,
                normalize=True,
                value_range=(-1, 1),
            )
            print(f"Saved combined video to {output_path}")
        else:
            # Just save the output video
            output_path = os.path.join(real_output_dir, "output.mp4")
            cache_video(
                tensor=decoded_video,
                save_file=output_path,
                fps=16,
                normalize=True,
                value_range=(-1, 1),
            )
            print(f"Saved output video to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo control LoRA test script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./models",
        help="Path to pretrained HunyuanVideo model directory",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        help="Bit depth for the base model, default config with nf4=16GB",
        choices=["nf4", "int8", "bf16"],
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to LoRA safetensors file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/test_control",
        help="Output directory for test results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Positive prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="./embeddings/default_empty_hyv.safetensors",
        help="Precalculated positive embedding file to use if no prompt is provided",
    )
    parser.add_argument(
        "--negative_embedding",
        type=str,
        default="./embeddings/default_video_negative_hyv.safetensors",
        help="Precalculated negative embedding file to use if no negative prompt is provided",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible generation",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated video",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated video",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=49,  # roughly 2 seconds, follows the VAE frame pattern
        help="Number of frames to generate (should follow VAE frame pattern of 4*n+1)",
    )
    parser.add_argument(
        "--control_video",
        type=str,
        default=None,
        help="Control video to use with control LoRA",
    )
    parser.add_argument(
        "--control_preprocess",
        type=str,
        default="none",
        choices=["none", "depth"],
        help="Preprocessing to apply to control video",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
