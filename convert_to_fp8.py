# NEEDS REFINEMENT, CURRENTLY NOT WORKING AS EXPECTED
import argparse
import os
import torch
from safetensors.torch import save_file
from transformers import CLIPTextModel, LlamaModel
from diffusers import HunyuanVideoTransformer3DModel
import gc
import shutil

try:
    import quanto
    print("Successfully imported quanto.")
except ImportError:
    print("ERROR: quanto library not found. Please install it: pip install quanto")
    exit()

# Use qfloat8_e4m3fn for weights; leave activations unquantized
FP8_WEIGHTS = quanto.qfloat8_e4m3fn
FP8_ACTIVATIONS = None

def convert_model_to_fp8(model_dir, model_subfolder, model_class, output_dir):
    """Loads a model, converts its parameters to FP8 using quanto, and saves it."""
    print(f"\n--- Converting {model_subfolder} to FP8 ---")
    input_path = os.path.join(model_dir, model_subfolder)
    output_path = os.path.join(output_dir, model_subfolder)
    os.makedirs(output_path, exist_ok=True)

    # Force all conversions to CPU to manage memory
    device = 'cpu'
    print(f"Using device: {device} for {model_subfolder} conversion.")

    # Load model
    print(f"Loading model from: {input_path}")
    try:
        original_dtype = torch.bfloat16
        model = model_class.from_pretrained(
            input_path if model_class is HunyuanVideoTransformer3DModel else model_dir,
            subfolder=model_subfolder if model_class is not HunyuanVideoTransformer3DModel else None,
            torch_dtype=original_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print(f"Loaded {model_subfolder}. Parameter count: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Error loading model {model_subfolder}: {e}")
        return

    # Apply quantization
    print(f"Quantizing to FP8 (weights={FP8_WEIGHTS}, activations={FP8_ACTIVATIONS})...")
    try:
        quanto.quantize(model, weights=FP8_WEIGHTS, activations=FP8_ACTIVATIONS)
        print("Quantization applied.")
    except Exception as e:
        print(f"Error during quantization: {e}")
        del model
        gc.collect()
        return

    # Verify quantization by checking module types
    print("Checking if modules are quantized:")
    quantized = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight') and isinstance(module.weight, quanto.QTensor):
                print(f"  {name}: Quantized with {module.weight.qtype}")
                quantized = True
            else:
                print(f"  {name}: Not quantized")
    if not quantized:
        print("Warning: No modules were quantized!")

    # Save quantized model
    quantized_state_dict = model.state_dict()
    output_file = os.path.join(output_path, "model.safetensors")
    print(f"Saving quantized model to: {output_file}")
    try:
        tensor_state_dict = {k: v.cpu() for k, v in quantized_state_dict.items() if isinstance(v, torch.Tensor)}
        save_file(tensor_state_dict, output_file)
        print("Quantized model saved.")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Handle config
    config_input = os.path.join(input_path, "config.json")
    config_output = os.path.join(output_path, "config.json")
    if os.path.exists(config_input):
        try:
            shutil.copyfile(config_input, config_output)
            print(f"Copied config to: {config_output}")
        except Exception as e:
            print(f"Failed to copy config: {e}")
            model.config.save_pretrained(output_path)
            print(f"Saved config from model: {config_output}")
    else:
        model.config.save_pretrained(output_path)
        print(f"Saved config from model: {config_output}")

    # Cleanup
    del model, quantized_state_dict, tensor_state_dict
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Convert models to FP8")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to original model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save FP8 models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_convert = {
        "transformer": HunyuanVideoTransformer3DModel,
        "text_encoder": LlamaModel,
        "text_encoder_2": CLIPTextModel,
    }

    for subfolder, model_class in models_to_convert.items():
        convert_model_to_fp8(args.model_dir, subfolder, model_class, args.output_dir)

    print("\n--- FP8 Conversion Complete ---")

if __name__ == "__main__":
    main()