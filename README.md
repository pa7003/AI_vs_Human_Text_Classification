# AI vs Human Text Classification

A small project for binary sentiment classification on the IMDB movie-reviews dataset using a transformer-based model. This repository contains training/evaluation scripts, model and tokenizer artifacts, and example logs from a training run.

## Summary

- Dataset: IMDB movie reviews (50,000 samples; 25,000 train / 25,000 test). Labels are balanced (0 / 1).
- Task: Binary sentiment classification (negative vs. positive).
- Frameworks used in example runs: Hugging Face Transformers & Datasets, PyTorch (training logs show PyTorch). Some logs include JAX/TPU-related warnings (informational only).

## Repository layout

- AI_vs_Human_classifier/ — model artifacts, training & evaluation scripts, and this README.
- Model / tokenizer files seen in logs:
  - `model.safetensors`
  - `vocab.txt`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `config.json`

## Quick results (example run)

Training progress (excerpt from run logs):

| Epoch | Training Loss | Validation Loss | F1       | Accuracy |
|-------:|--------------:|----------------:|---------:|---------:|
| 1      | 0.361000      | 0.367621        | 0.846709 | 0.847346 |
| 2      | 0.314700      | 0.326106        | 0.869130 | 0.862237 |
| 3      | 0.220800      | 0.360525        | 0.876580 | 0.870442 |
| 4      | 0.142100      | 0.417613        | 0.878367 | 0.873278 |

Final evaluation metrics (after epoch 4):

- eval_loss: 0.41761314868927  
- eval_f1: 0.8783665532328634  
- eval_accuracy: 0.8732779578606159  
- eval_runtime: 9.8025s

Additional evaluation metrics (from another evaluation run / log):

{'eval_loss': 0.41761314868927, 'eval_f1': 0.8783665532328634, 'eval_accuracy': 0.8732779578606159, 'eval_runtime': 10.0107, 'eval_samples_per_second': 1000.529, 'eval_steps_per_second': 31.267, 'epoch': 4.0}

These values come from example training and evaluation logs included with the repository.

## Notable logs & warnings (relevant items from run)

- Dataset download / generation:
  - Dataset loaded from Hugging Face Hub: `imdb`
  - Generated splits: train (25k), test (25k), unsupervised (50k)
  - Dataset size: 50,000 samples; balanced label distribution (25,000 / 25,000)
- Environment / runtime warnings:
  - Jupyter/Colab: If running in a notebook, you may need to restart the Python kernel/runtime after installing packages to fully load dependencies.
  - torch_xla SyntaxWarning: e.g. invalid escape sequence warning from `torch_xla` (informational).
  - JAX/TPU warning: "Transparent hugepages are not enabled..." — only relevant if you plan to run on TPU.
  - PyTorch DataLoader warning: "'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used." — indicates `pin_memory=True` while no GPU/accelerator was detected; set `pin_memory=False` or run on a GPU for better performance.
- Authentication:
  - Logs show "Logged in to Hugging Face Hub." If you use private models/datasets, ensure HF credentials are configured.

## How to run (example commands)

Adjust script names and arguments to match the actual scripts in `AI_vs_Human_classifier/`.

1. Install dependencies (example):
   ```
   python -m pip install -r requirements.txt
   ```

2. Train (example):
   ```
   python train.py --dataset imdb --model_name_or_path ./pretrained-model --output_dir ./runs/run1
   ```

3. Evaluate (example):
   ```
   python evaluate.py --model_dir ./runs/run1 --dataset imdb
   ```

4. Inference (example):
   ```
   python predict.py --model_dir ./runs/run1 --text "This movie was fantastic!"
   ```

Notes:
- If you run in Colab/Jupyter, restart the kernel after installing packages if you see missing imports or unexpected behavior.
- If you have a GPU, ensure the scripts use `device="cuda"` and set `pin_memory=True`. If running on CPU, set `pin_memory=False` to avoid DataLoader warnings.

## Files & artifacts

- Tokenizer files: `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`
- Model weights: `model.safetensors`
- Model config: `config.json`
- Training log excerpts and dataset download logs are useful for debugging and reproducibility.

## Reproducibility & environment

- The example logs indicate dataset downloads from Hugging Face and model/tokenizer files being loaded. Ensure:
  - Internet access for dataset/model downloads (or pre-download artifacts).
  - Hugging Face credentials if you need access to private resources.
- Recommendation: add a `requirements.txt` with pinned package versions that were used to produce the example logs.

## Suggestions / TODO

- Add a `requirements.txt` with pinned versions used for the example run.
- Add (or document) concrete CLI scripts or a Jupyter notebook demonstrating:
  - Full training loop
  - Evaluation and metric computation
  - Inference example
- Document how to enable GPU/TPU training and recommended settings (batch size, `pin_memory`, mixed precision).
- Add tests or a small example to quickly verify the environment (e.g., a tiny dataset run).

## License

Add a `LICENSE` file (e.g., MIT, Apache-2.0) if you plan to make the repository public.

## Contact

Maintainer: pa7003
