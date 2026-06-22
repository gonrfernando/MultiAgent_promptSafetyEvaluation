<<<<<<< HEAD
# MultiAgent_promptSafetyEvaluation
=======
# Can Agent Disagreement Predict Unsafe LLM Outputs Across Languages?

Research pipeline investigating whether disagreement among LLM judges
can signal unsafe prompts across Spanish variants (neutral and regional slang).

#This repo has two branches, one for each pipeline mentioned in our report

## Setup

```powershell
cd C:\Users\gonrf\Projects\agent-disagreement-safety
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

Ensure Ollama is running and models are pulled:

```powershell
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
```

## 1. Import prompts from prompts.txt

`prompts.txt` must contain blocks of 6 lines in this order:

1. es_neutral
2. es_mx_gdl
3. es_ar_caba
4. es_cl_santiago
5. es_es_bcn
6. es_uy

No `category` or `expected_risk` is stored, to avoid judge bias.

```powershell
python -m src.pipeline.import_prompts_txt
```

Output: `data/prompts/processed/prompts.jsonl`

## 2. Evaluate prompts with 3 Ollama judges

Judges are configured in `configs/judges.yaml`:

| Judge | Model | Focus |
|-------|-------|-------|
| security | llama3.1:8b | physical harm, weapons, drugs |
| policy | mistral:7b | TOS, fraud, prohibited content |
| critical | qwen2.5:7b | hidden intent, slang, edge cases |

Pilot with the first 12 prompts (2 blocks):

```powershell
python -m src.pipeline.evaluate_prompts --run-id pilot-2blocks --limit 12
```

Full run (120 prompts, 360 evaluations):

```powershell
python -m src.pipeline.evaluate_prompts --run-id judges-full
```

Outputs:

- `data/evaluations/{run_id}/evaluations.jsonl`
- `data/evaluations/{run_id}/disagreement_metrics.jsonl`
- `data/evaluations/{run_id}/run_metadata.json`

Judges receive **only the prompt text**, never category or expected risk.

## Analyze full run (charts)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv\Scripts\python.exe analyze.py
```

Optional:

```powershell
.\.venv\Scripts\python.exe analyze.py --run-dir data/evaluations/judges-full
```

Figures and CSV summaries go to `data/evaluations/judges-full/figures/`.

## Tests

```powershell
pytest
```

## License

Research / hackathon use.
>>>>>>> ef4a49c (Working pipeline)
