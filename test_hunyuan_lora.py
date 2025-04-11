import os
import gc
import argparse
import torch
import bitsandbytes as bnb

from peft import PeftModel, LoraConfig, set_peft_model_state_dict

from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
from safetensors.torch import load_file, save_file

from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.utils import export_to_video


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo lora test script",
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
        type = str,
        default = None,
        help = "LoRA file to test",
        )
    parser.add_argument(
        "--alpha",
        type = int,
        default = None,
        help = "lora alpha, defaults to 1"
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./test/test_lora",
        help = "Output directory for results"
        )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for inference"
        )
    parser.add_argument(
        "--width",
        type = int,
        default = 512,
        help = "Width for inference"
        )
    parser.add_argument(
        "--height",
        type = int,
        default = 512,
        help = "Width for inference"
        )
    parser.add_argument(
        "--num_frames",
        type = int,
        default = 33,
        help = "Number of frames per video, must be divisible by 4+1"
        )
    parser.add_argument(
        "--inference_steps",
        type = int,
        default = 20,
        help = "Number of steps for inference",
        )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person typing on a laptop keyboard",
        help="Prompt for inference",
        )
    
    args = parser.parse_args()
    return args

@torch.inference_mode()
def main(args):
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model,
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(args.pretrained_model, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling(
        tile_sample_min_height = 256,
        tile_sample_min_width = 256,
        tile_sample_min_num_frames = 64,
        tile_sample_stride_height = 192,
        tile_sample_stride_width = 192,
        tile_sample_stride_num_frames = 16,
    )
    pipe.enable_sequential_cpu_offload()

    output = pipe(
        prompt = args.prompt,
        height = args.height,
        width = args.width,
        num_frames = args.num_frames,
        num_inference_steps = args.inference_steps,
        generator = torch.Generator(device="cpu").manual_seed(args.seed),
    ).frames[0]
    
    export_to_video(
        output,
        os.path.join(args.output_dir, "output_base.mp4"),
        fps=15,
    )
    
    if args.lora is not None:
        del transformer
        pipe.transformer = None
        gc.collect()
        torch.cuda.empty_cache()
        
        lora_sd = load_file(args.lora)
        rank = 0
        for key in lora_sd.keys():
            if ".lora_A.weight" in key:
                rank = lora_sd[key].shape[0]
        
        alpha = 1 if args.alpha is None else args.alpha
        lora_weight = alpha / rank
        
        print(f"lora rank = {rank}")
        print(f"alpha = {alpha}")
        print(f"lora weight = {lora_weight}")
        
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder = "transformer",
            torch_dtype = torch.bfloat16,
        )
        
        transformer.load_lora_adapter(lora_sd, adapter_name="default_lora")
        
        transformer.set_adapters(adapter_names = "default_lora", weights = lora_weight)
        pipe.transformer = transformer
        pipe.enable_sequential_cpu_offload()
        
        output_lora = pipe(
            prompt = args.prompt,
            height = args.height,
            width = args.width,
            num_frames = args.num_frames,
            num_inference_steps = args.inference_steps,
            generator = torch.Generator(device="cpu").manual_seed(args.seed),
        ).frames[0]
        
        export_to_video(
            output_lora,
            os.path.join(args.output_dir, "output_lora.mp4"),
            fps=15,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)