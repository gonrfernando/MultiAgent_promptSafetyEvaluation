# Multi-Agent Prompt Safety Evaluation

This repository contains a small benchmark setup for evaluating how open-source
chat models respond to safety-sensitive prompts. The project was organized into
separate, model-specific runners so each model can be tested independently and
the generated outputs can be reviewed in CSV format.

## Project Structure

```text
.
|-- run_phi3.py             # Benchmark runner for Microsoft Phi-3 Mini
|-- run_llama2.py           # Benchmark runner for Meta Llama 2 Chat
|-- run_mistral.py          # Benchmark runner for Mistral 7B Instruct
|-- prompts.txt             # Input prompts, one prompt per line
|-- results_phi3.csv        # Phi-3 benchmark outputs
|-- results_llama2.csv      # Llama 2 benchmark outputs
|-- results_mistral.csv     # Mistral benchmark outputs
|-- results_uruguay.csv     # Previous Uruguay-focused benchmark outputs
|-- test_uruguay.py         # Legacy combined runner used during experimentation
`-- data/                   # Supporting benchmark behavior datasets
```

## Models

The benchmark currently supports three Hugging Face models:

- `microsoft/Phi-3-mini-4k-instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`

Each model has its own script to keep the evaluation workflow clear and easy to
reproduce.

## Environment Setup

This project expects a Python environment with CUDA-capable PyTorch installed.
The scripts use 4-bit quantization to reduce GPU memory requirements.

Install the main dependencies:

```bash
pip install torch transformers accelerate bitsandbytes
```

For Llama 2, you must also request access to the model on Hugging Face and set a
token in your environment:

```bash
export HF_TOKEN="your_hugging_face_token"
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

## Running the Benchmarks

Run each model independently:

```bash
python run_phi3.py
python run_llama2.py
python run_mistral.py
```

Use `--limit` for a quick smoke test:

```bash
python run_phi3.py --limit 5
```

Resume from a specific prompt ID and append to an existing results file:

```bash
python run_llama2.py --start-at-prompt-id 49 --append-results
```

Use a custom prompt file or output file:

```bash
python run_mistral.py --prompts-file prompts.txt --results-file results_mistral.csv
```

## Output Format

Each runner writes a CSV file with the following columns:

- `timestamp`: UTC timestamp for the generation attempt.
- `model`: Model key used for the run.
- `prompt_id`: Line number from the prompt file.
- `prompt`: Original prompt text.
- `response`: Generated model response.
- `latency_sec`: Generation latency in seconds.
- `error`: Error message, if model loading or generation failed.

## Notes for Reviewers

The current version focuses on reproducible local inference rather than hosted
API evaluation. Each runner performs basic GPU diagnostics, loads the model in
4-bit mode, applies the model chat template, generates responses, and records
both successful outputs and failures in CSV format.

The scripts are intentionally simple and self-contained so reviewers can inspect
the benchmark logic without following cross-file abstractions.
