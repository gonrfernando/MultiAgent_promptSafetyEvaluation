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
PROMPTS_FILE = SCRIPT_DIR / "prompts_uruguay.txt"
RESULTS_FILE = SCRIPT_DIR / "results_uruguay.csv"

MODELS = {
    "phi3": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": False,
        "min_vram_gb": 3.0,
        "needs_hf_token": False,
    },
    "llama2": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        "trust_remote_code": False,
        "min_vram_gb": 4.5,
        "needs_hf_token": True,
    },
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

START_AT_PROMPT_ID = 1


def setup_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_hf_token_kwargs():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append({"prompt_id": line_idx, "prompt": prompt})
    return prompts


def get_gpu_vram_gb():
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def get_quant_config(cpu_offload: bool = False):
    config_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    }
    if cpu_offload:
        config_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**config_kwargs)


def get_model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return model.get_input_embeddings().weight.device


def load_model(model_key: str):
    cfg = MODELS[model_key]
    token_kwargs = get_hf_token_kwargs()
    vram_total_gb = get_gpu_vram_gb()
    min_vram_gb = cfg.get("min_vram_gb", 4.0)
    needs_offload = vram_total_gb < min_vram_gb

    print(f"\nCargando {model_key} ({cfg['hf_id']}) en 4-bit...")
    if needs_offload:
        gpu_budget_gb = max(round(vram_total_gb - 0.3, 1), 3.0)
        print(
            f"VRAM limitada ({vram_total_gb:.1f} GB < {min_vram_gb} GB): "
            f"4-bit con reparto GPU/CPU (~{gpu_budget_gb} GB en GPU)."
        )
    else:
        print(f"VRAM suficiente ({vram_total_gb:.1f} GB): modelo completo en GPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["hf_id"],
        trust_remote_code=cfg["trust_remote_code"],
        **token_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "quantization_config": get_quant_config(cpu_offload=needs_offload),
        "low_cpu_mem_usage": True,
        "trust_remote_code": cfg["trust_remote_code"],
        "attn_implementation": "eager",
        **token_kwargs,
    }
    if needs_offload:
        gpu_budget_gb = max(round(vram_total_gb - 0.3, 1), 3.0)
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = {0: f"{gpu_budget_gb}GiB", "cpu": "14GiB"}
    else:
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **model_kwargs)

    vram_gb = torch.cuda.memory_allocated() / 1e9
    has_cpu_layers = any(param.device.type == "cpu" for param in model.parameters())
    offload_note = " (parte en CPU — sera lento)" if has_cpu_layers else " (100% GPU)"
    print(f"Modelo {model_key} listo. VRAM usada: {vram_gb:.2f} GB{offload_note}")
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_response(model, tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

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

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    latency_sec = start.elapsed_time(end) / 1000
    return response, latency_sec


def append_result(path: Path, row: dict, write_header: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def filter_prompts_from_id(prompts, start_at_prompt_id: int):
    if start_at_prompt_id <= 0:
        return prompts
    filtered = [p for p in prompts if p["prompt_id"] >= start_at_prompt_id]
    if not filtered:
        raise ValueError(
            f"No hay prompts con prompt_id >= {start_at_prompt_id}. "
            f"Rango disponible: 1-{prompts[-1]['prompt_id']}"
        )
    return filtered


def run_benchmark(prompts, model_keys, results_path: Path, resume: bool = False):
    if resume:
        write_header = not results_path.exists() or results_path.stat().st_size == 0
    else:
        results_path.unlink(missing_ok=True)
        write_header = True
    total = len(prompts) * len(model_keys)
    test_number = 0

    for model_key in model_keys:
        model = None
        tokenizer = None
        try:
            model, tokenizer = load_model(model_key)
        except Exception as exc:
            print(f"No se pudo cargar {model_key}: {exc}")
            traceback.print_exc()
            for item in prompts:
                append_result(
                    results_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model_key,
                        "prompt_id": item["prompt_id"],
                        "prompt": item["prompt"],
                        "response": "",
                        "latency_sec": "",
                        "error": f"model_load_failed: {exc}",
                    },
                    write_header,
                )
                write_header = False
            continue

        for item in prompts:
            test_number += 1
            print(f"[{test_number}/{total}] {model_key} | prompt {item['prompt_id']}")
            print(f"  Prompt: {item['prompt'][:100]}...")

            try:
                response, latency_sec = generate_response(
                    model, tokenizer, item["prompt"]
                )
                error = ""
                print(f"  Tiempo: {latency_sec:.1f}s")
                print(f"  Respuesta: {response[:160]}...")
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
                    "model": model_key,
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

        unload_model(model, tokenizer)


def print_diagnostics():
    print("=" * 60)
    print("DIAGNOSTICO")
    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No se detecto GPU")
        sys.exit(1)

    print("GPU:", torch.cuda.get_device_name(0))
    vram_total = get_gpu_vram_gb()
    print("VRAM total:", round(vram_total, 2), "GB")
    for key, cfg in MODELS.items():
        if vram_total < cfg["min_vram_gb"]:
            print(
                f"Aviso: {key} puede necesitar CPU offload en esta GPU "
                f"({vram_total:.1f} GB < {cfg['min_vram_gb']} GB)."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prueba prompts uruguayos con Phi-3 y Llama-2."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=PROMPTS_FILE,
        help="Archivo de prompts (un prompt por linea).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help="CSV de salida con respuestas.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Modelos a evaluar (orden: phi3, luego llama2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limitar cantidad de prompts (0 = todos).",
    )
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()
    print_diagnostics()

    if not args.prompts_file.exists():
        print(f"No existe el archivo de prompts: {args.prompts_file}")
        sys.exit(1)

    prompts = load_prompts(args.prompts_file)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    if not prompts:
        print(f"Error: no se encontraron prompts en {args.prompts_file}")
        print("Verifica que el archivo tenga un prompt por linea.")
        sys.exit(1)

    resume = START_AT_PROMPT_ID > 0
    if resume:
        try:
            prompts = filter_prompts_from_id(prompts, START_AT_PROMPT_ID)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(
            f"\nReanudando desde prompt_id {START_AT_PROMPT_ID} "
            f"({len(prompts)} prompts restantes)"
        )

    print(f"\nPrompts cargados: {len(prompts)}")
    print(f"Modelos: {', '.join(args.models)}")
    print(f"Total de pruebas: {len(prompts) * len(args.models)}")
    print(f"Resultados en: {args.results_file}")

    models_needing_token = [
        key for key in args.models if MODELS[key].get("needs_hf_token")
    ]
    if models_needing_token and not get_hf_token_kwargs():
        print(
            "\nAviso: "
            + ", ".join(models_needing_token)
            + " requiere HF_TOKEN y aceptar la licencia en Hugging Face."
        )

    print("\n" + "=" * 60)
    print("INICIANDO BENCHMARK")
    print("=" * 60)

    run_benchmark(
        prompts=prompts,
        model_keys=args.models,
        results_path=args.results_file,
        resume=resume,
    )

    print("\nBenchmark completado.")
    print(f"Revisa {args.results_file}")


if __name__ == "__main__":
    main()
