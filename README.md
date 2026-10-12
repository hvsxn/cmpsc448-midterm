# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project - Nihat Hasan (individual participant and team leader).

## Main findings

On 363 held-out responses from 121 unseen prompts, the output-only CNN achieved
70.25% accuracy and the output-only LSTM achieved 49.86%. Prompt-only accuracy
was exactly 33.33% for both models because each prompt occurs once per class.
Adding prompts changed CNN accuracy to 65.84% and LSTM accuracy to 51.24%.

Read `Midterm_Report.pdf` for the full methods, results, limitations, and analysis.

## Reproduce

Use Python with the packages in requirements.txt:

```bash
pip install -r requirements.txt
python main.py
```

The script downloads the public AlpacaEval response file and runs six experiments.
GPU is used if available; CPU is supported but slower. Results go in `run_results/`.
`original_colab_run.py` is the exact script saved during the submitted Colab run.
`main.py` changes only output paths and the optional Colab download for portability.
The original experiment ran on CUDA; the GPU model and package versions were not
recorded. Seed 42 is set, but bitwise reproduction across environments is not guaranteed.

## Data and evaluation

Source: https://huggingface.co/datasets/tatsu-lab/alpaca_eval

Download: https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_all_outputs.json

Upstream project: https://github.com/tatsu-lab/alpaca_eval

Models: `gpt4`, `claude`, and `vicuna-13b` (LLaMA-derived).
There are 805 shared prompts and 2,415 responses. Prompt-grouped splitting gives
563 train, 121 validation, and 121 test prompts, each with three responses.
The train-only vocabulary contains 12,003 entries. Best validation macro-F1
selects a checkpoint before test evaluation. All six configurations were specified
before examining test results; no further model tuning was performed afterward.

The repository does not redistribute the raw response dataset. Obtain it from the
source and consult its upstream usage terms. Prompt hashes and predicted labels are
included for checking results. The local dataset checksum is in dataset_sha256.txt.
The download uses the upstream main branch, so future source changes may affect reruns.

## Files

- Midterm_Report.pdf: project report.
- main.py: portable training and evaluation script.
- original_colab_run.py: original Colab code.
- results.csv, training_history.csv, metadata.json: original experiment records.
- *_predictions.csv: test labels and SHA-256 prompt identifiers.
- *_confusion.png: original confusion matrices.
- training_text_statistics.csv: descriptive training-data statistics.
- analyze.py and paired_bootstrap.json: prompt-level paired bootstrap analysis.

Run `python analyze.py` to recompute the paired bootstrap from saved predictions.
Model checkpoints and full local data remain in the original results archive;
they are unnecessary for inspecting the reported metrics or rerunning training.

## Limitations and AI assistance

One seed, one dataset, older model snapshots, truncation, and minimal tuning limit
generalization. Newline statistics are descriptive: the tokenizer discards whitespace,
so those counts are not evidence that the classifiers directly use newlines.
No cross-domain experiment or causal feature-removal ablation was performed.

ChatGPT assisted with code drafting, debugging guidance, analysis, and report drafting.
The experiments were run in Google Colab. The author is responsible for reviewing
and explaining the implementation and submitted results.
