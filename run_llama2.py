"""Run a prompt benchmark with Meta Llama 2 Chat.

The script reads one prompt per line from a text file, generates one response per
prompt, and stores the results in a CSV file that is easy to inspect or submit.
"""

import argparse
import csv
import gc
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_FILE = SCRIPT_DIR / "prompts.txt"
DEFAULT_RESULTS_FILE = SCRIPT_DIR / "results_llama2.csv"

MODEL_CONFIG = {
    "name": "llama2",
    "hf_id": "meta-llama/Llama-2-7b-chat-hf",
    "trust_remote_code": False,
    "min_vram_gb": 4.5,
    "requires_hf_token": True,
}

GENERATION_KWARGS = {
    "max_new_tokens": 300,
    "temperature": 0.9,
    "do_sample": True,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}

RESULT_FIELDS = (
    "timestamp",
    "model",
    "prompt_id",
    "prompt",
    "response",
    "latency_sec",
    "error",
)


def setup_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_hf_token_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def load_prompts(path: Path) -> list[dict[str, str | int]]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            prompt = line.strip()
            if prompt:
                prompts.append({"prompt_id": line_number, "prompt": prompt})
    return prompts


def get_gpu_vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def get_quantization_config(cpu_offload: bool) -> BitsAndBytesConfig:
    config_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    }
    if cpu_offload:
        config_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**config_kwargs)


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return model.get_input_embeddings().weight.device


def load_model():
    token_kwargs = get_hf_token_kwargs()
    vram_total_gb = get_gpu_vram_gb()
    min_vram_gb = MODEL_CONFIG["min_vram_gb"]
    needs_offload = vram_total_gb < min_vram_gb

    print(f"\nLoading {MODEL_CONFIG['name']} ({MODEL_CONFIG['hf_id']}) in 4-bit mode...")
    if needs_offload:
        gpu_budget_gb = max(round(vram_total_gb - 0.3, 1), 3.0)
        print(
            f"Limited VRAM ({vram_total_gb:.1f} GB < {min_vram_gb} GB): "
            f"using GPU/CPU offload with about {gpu_budget_gb} GB on GPU."
        )
    else:
        print(f"Enough VRAM detected ({vram_total_gb:.1f} GB): loading on GPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["hf_id"],
        trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        **token_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "quantization_config": get_quantization_config(cpu_offload=needs_offload),
        "low_cpu_mem_usage": True,
        "trust_remote_code": MODEL_CONFIG["trust_remote_code"],
        "attn_implementation": "eager",
        **token_kwargs,
    }
    if needs_offload:
        gpu_budget_gb = max(round(vram_total_gb - 0.3, 1), 3.0)
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = {0: f"{gpu_budget_gb}GiB", "cpu": "14GiB"}
    else:
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["hf_id"], **model_kwargs)
    vram_used_gb = torch.cuda.memory_allocated() / 1e9
    has_cpu_layers = any(param.device.type == "cpu" for param in model.parameters())
    offload_note = "CPU offload enabled" if has_cpu_layers else "fully on GPU"
    print(f"Model ready. VRAM used: {vram_used_gb:.2f} GB ({offload_note}).")
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_response(model, tokenizer, prompt: str) -> tuple[str, float]:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_KWARGS,
        )
    end.record()
    torch.cuda.synchronize()

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    latency_sec = start.elapsed_time(end) / 1000
    return response, latency_sec


def append_result(path: Path, row: dict, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def filter_prompts_from_id(
    prompts: list[dict[str, str | int]],
    start_at_prompt_id: int,
) -> list[dict[str, str | int]]:
    if start_at_prompt_id <= 1:
        return prompts

    filtered = [prompt for prompt in prompts if prompt["prompt_id"] >= start_at_prompt_id]
    if not filtered:
        raise ValueError(
            f"No prompts found with prompt_id >= {start_at_prompt_id}. "
            f"Available range: 1-{prompts[-1]['prompt_id']}."
        )
    return filtered


def run_benchmark(
    prompts: list[dict[str, str | int]],
    results_path: Path,
    append_results: bool,
) -> None:
    if append_results:
        write_header = not results_path.exists() or results_path.stat().st_size == 0
    else:
        results_path.unlink(missing_ok=True)
        write_header = True

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model()
        for test_number, item in enumerate(prompts, start=1):
            print(
                f"[{test_number}/{len(prompts)}] "
                f"{MODEL_CONFIG['name']} | prompt {item['prompt_id']}"
            )
            print(f"  Prompt: {str(item['prompt'])[:100]}...")

            try:
                response, latency_sec = generate_response(
                    model,
                    tokenizer,
                    str(item["prompt"]),
                )
                error = ""
                print(f"  Time: {latency_sec:.1f}s")
                print(f"  Response: {response[:160]}...")
            except Exception as exc:
                response = ""
                latency_sec = ""
                error = str(exc)
                print(f"  Error: {exc}")
                traceback.print_exc()

            append_result(
                results_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL_CONFIG["name"],
                    "prompt_id": item["prompt_id"],
                    "prompt": item["prompt"],
                    "response": response,
                    "latency_sec": latency_sec,
                    "error": error,
                },
                write_header,
            )
            write_header = False
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"Could not load {MODEL_CONFIG['name']}: {exc}")
        traceback.print_exc()
        for item in prompts:
            append_result(
                results_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL_CONFIG["name"],
                    "prompt_id": item["prompt_id"],
                    "prompt": item["prompt"],
                    "response": "",
                    "latency_sec": "",
                    "error": f"model_load_failed: {exc}",
                },
                write_header,
            )
            write_header = False
    finally:
        if model is not None and tokenizer is not None:
            unload_model(model, tokenizer)


def print_diagnostics() -> None:
    print("=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No CUDA GPU was detected. This benchmark requires a CUDA-capable GPU.")
        sys.exit(1)

    print("GPU:", torch.cuda.get_device_name(0))
    vram_total_gb = get_gpu_vram_gb()
    print("Total VRAM:", round(vram_total_gb, 2), "GB")
    if vram_total_gb < MODEL_CONFIG["min_vram_gb"]:
        print(
            f"Warning: {MODEL_CONFIG['name']} may need CPU offload on this GPU "
            f"({vram_total_gb:.1f} GB < {MODEL_CONFIG['min_vram_gb']} GB)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prompt benchmark with Meta Llama 2 Chat."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="Text file with one prompt per line.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Output CSV file for generated responses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of prompts to run. Use 0 for all prompts.",
    )
    parser.add_argument(
        "--start-at-prompt-id",
        type=int,
        default=1,
        help="Start from this prompt_id. Useful for resuming interrupted runs.",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append to the results CSV instead of replacing it.",
    )
    return parser.parse_args()


def main() -> None:
    setup_console()
    args = parse_args()
    print_diagnostics()

    if not args.prompts_file.exists():
        print(f"Prompt file not found: {args.prompts_file}")
        sys.exit(1)

    prompts = load_prompts(args.prompts_file)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    if not prompts:
        print(f"No prompts were found in {args.prompts_file}.")
        print("Make sure the file contains one prompt per line.")
        sys.exit(1)

    try:
        prompts = filter_prompts_from_id(prompts, args.start_at_prompt_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    append_results = args.append_results or args.start_at_prompt_id > 1
    print(f"\nLoaded prompts: {len(prompts)}")
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Results file: {args.results_file}")
    if append_results:
        print("Results will be appended.")

    if MODEL_CONFIG["requires_hf_token"] and not get_hf_token_kwargs():
        print(
            "\nWarning: this model requires HF_TOKEN and license access on Hugging Face."
        )

    print("\n" + "=" * 60)
    print("STARTING BENCHMARK")
    print("=" * 60)

    run_benchmark(
        prompts=prompts,
        results_path=args.results_file,
        append_results=append_results,
    )

    print("\nBenchmark complete.")
    print(f"Review the results in {args.results_file}")


if __name__ == "__main__":
    main()
