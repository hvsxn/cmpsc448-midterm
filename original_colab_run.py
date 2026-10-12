import os, json, random, re, time, copy, zipfile
from pathlib import Path
from collections import Counter

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

START = time.time()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

OUT = Path("/content/midterm_results")
OUT.mkdir(exist_ok=True)

# save cell as a reproducible Python script
try:
    (OUT / "main.py").write_text(In[-1], encoding="utf-8")
except Exception:
    pass

# public, precomputed AlpacaEval responses
URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/"
    "resolve/main/alpaca_eval_all_outputs.json"
)
r = requests.get(URL, timeout=90)
r.raise_for_status()
raw = r.json()
df = pd.DataFrame(raw)
print("Available models:", sorted(df["generator"].unique()), flush=True)

# 3 different families: OpenAI GPT, Anthropic Claude,
# & Vicuna, a model fine-tuned from Meta's LLaMA
MODELS = ["gpt4", "claude", "vicuna-13b"]
missing = set(MODELS) - set(df["generator"])
if missing:
    raise ValueError(f"Missing models: {missing}. Send the available-model list.")

df = df[df["generator"].isin(MODELS)].copy()
df = df.dropna(subset=["instruction", "output"])
df["instruction"] = df["instruction"].astype(str).str.strip()
df["output"] = df["output"].astype(str).str.strip()
df = df[(df["instruction"] != "") & (df["output"] != "")]
df = df.drop_duplicates(["instruction", "generator"])

# keep only prompts answered by all 3 models
counts = df.groupby("instruction")["generator"].nunique()
prompts = sorted(counts[counts == 3].index)
assert len(prompts) >= 100, f"Only {len(prompts)} shared prompts."
df = df[df["instruction"].isin(prompts)].copy()
label_map = {name: i for i, name in enumerate(MODELS)}
df["label"] = df["generator"].map(label_map)

# split by PROMPT, so responses to one prompt never cross splits
train_p, rest_p = train_test_split(
    prompts, test_size=0.30, random_state=SEED
)
val_p, test_p = train_test_split(
    rest_p, test_size=0.50, random_state=SEED
)
split_map = {
    **dict.fromkeys(train_p, "train"),
    **dict.fromkeys(val_p, "validation"),
    **dict.fromkeys(test_p, "test")
}
df["split"] = df["instruction"].map(split_map)
df = df.sort_values(["instruction", "generator"]).reset_index(drop=True)

print("\nExamples per split and model:")
print(pd.crosstab(df["split"], df["generator"]), flush=True)

# vocab is built ONLY from training text
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

train_df = df[df["split"] == "train"]
counter = Counter()
for row in train_df.itertuples():
    counter.update(tokenize(row.instruction))
    counter.update(tokenize(row.output))

vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
for word, _ in counter.most_common(12000):
    if word not in vocab:
        vocab[word] = len(vocab)

PROMPT_LIMIT = 96
RESPONSE_LIMIT = 160
BATCH = 64
EPOCHS = 8

def encode(row, mode):
    p = [vocab.get(t, 1) for t in tokenize(row.instruction)[:PROMPT_LIMIT]]
    o = [vocab.get(t, 1) for t in tokenize(row.output)[:RESPONSE_LIMIT]]
    if mode == "input_only":
        ids, width = p, PROMPT_LIMIT
    elif mode == "output_only":
        ids, width = o, RESPONSE_LIMIT
    else:
        ids, width = p + [2] + o, PROMPT_LIMIT + 1 + RESPONSE_LIMIT
    ids = ids or [1]
    length = len(ids)
    return ids + [0] * (width - length), length

def make_loader(frame, mode, shuffle=False):
    pairs = [encode(row, mode) for row in frame.itertuples()]
    x = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    lengths = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    y = torch.tensor(frame["label"].values, dtype=torch.long)
    return DataLoader(
        TensorDataset(x, lengths, y),
        batch_size=BATCH, shuffle=shuffle
    )

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 64, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(64, 64, k) for k in (3, 4, 5)
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(192, 3)

    def forward(self, x, lengths):
        e = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            h = torch.relu(conv(e))
            # exclude windows consisting entirely of padding
            valid = (
                torch.arange(h.size(2), device=x.device)[None, :]
                < lengths[:, None]
            )
            h = h.masked_fill(~valid[:, None, :], -1e9)
            features.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(features, dim=1)))

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 64, padding_idx=0)
        self.rnn = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 3)

    def forward(self, x, lengths):
        h, _ = self.rnn(self.embedding(x))
        # select the final token, before padding
        last = h[torch.arange(x.size(0), device=x.device), lengths - 1]
        return self.fc(self.dropout(last))

def evaluate(model, loader):
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x.to(device), lengths.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    return (
        accuracy_score(truth, predictions),
        f1_score(truth, predictions, average="macro", zero_division=0),
        truth, predictions
    )

results, histories = [], []
test_df = df[df["split"] == "test"]
val_df = df[df["split"] == "validation"]

for mode in ["input_only", "output_only", "input_output"]:
    train_loader = make_loader(train_df, mode, True)
    val_loader = make_loader(val_df, mode)
    test_loader = make_loader(test_df, mode)

    for name, cls in [("CNN", CNN), ("LSTM", LSTM)]:
        torch.manual_seed(SEED)
        model = cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_f1, best_state, best_epoch = -1, None, None

        print(f"\nTraining {name}: {mode}", flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, n = 0.0, 0
            for x, lengths, y in train_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x, lengths), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
                n += len(y)

            va, vf, _, _ = evaluate(model, val_loader)
            histories.append({
                "model": name, "setting": mode, "epoch": epoch,
                "train_loss": total_loss / n,
                "validation_accuracy": va, "validation_macro_f1": vf
            })
            if vf > best_f1:
                best_f1 = vf
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            print(
                f"  Epoch {epoch}: loss={total_loss/n:.3f}, "
                f"val accuracy={va:.3f}, val F1={vf:.3f}",
                flush=True
            )

        # test only after selecting the checkpoint using validation
        model.load_state_dict(best_state)
        acc, f1, truth, pred = evaluate(model, test_loader)
        results.append({
            "model": name, "setting": mode,
            "test_accuracy": acc, "test_macro_f1": f1,
            "selected_epoch": best_epoch
        })
        tag = f"{name}_{mode}"
        torch.save(best_state, OUT / f"{tag}.pt")
        pd.DataFrame({
            "instruction": test_df["instruction"].values,
            "true_model": [MODELS[i] for i in truth],
            "predicted_model": [MODELS[i] for i in pred]
        }).to_csv(OUT / f"{tag}_predictions.csv", index=False)

        cm = confusion_matrix(truth, pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set(
            xticks=range(3), yticks=range(3),
            xticklabels=MODELS, yticklabels=MODELS,
            xlabel="Predicted model", ylabel="True model",
            title=f"{name}: {mode}"
        )
        fig.tight_layout()
        fig.savefig(OUT / f"{tag}_confusion.png", dpi=160)
        plt.close(fig)
        print(f"TEST: accuracy={acc:.4f}, macro-F1={f1:.4f}", flush=True)

        # save incrementally in case the session disconnects
        pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
        pd.DataFrame(histories).to_csv(OUT / "training_history.csv", index=False)

# descriptive analysis uses training data
stats = train_df.copy()
stats["response_words"] = stats["output"].str.split().str.len()
stats["response_characters"] = stats["output"].str.len()
stats["newline_count"] = stats["output"].str.count("\n")
stats["type_token_ratio"] = stats["output"].apply(
    lambda s: len(set(tokenize(s))) / max(1, len(tokenize(s)))
)
summary = stats.groupby("generator")[
    ["response_words", "response_characters", "newline_count", "type_token_ratio"]
].mean()
summary.to_csv(OUT / "training_text_statistics.csv")

df.to_csv(OUT / "dataset_with_splits.csv", index=False)
(OUT / "vocabulary.json").write_text(json.dumps(vocab), encoding="utf-8")
metadata = {
    "source": URL,
    "models": MODELS,
    "label_map": label_map,
    "seed": SEED,
    "shared_prompts": len(prompts),
    "total_examples": len(df),
    "split_examples": df["split"].value_counts().to_dict(),
    "vocabulary_size": len(vocab),
    "embedding_dimension": 64,
    "cnn_filters_per_kernel": 64,
    "cnn_kernel_sizes": [3, 4, 5],
    "lstm_hidden_size": 64,
    "dropout": 0.3,
    "batch_size": BATCH,
    "epochs": EPOCHS,
    "learning_rate": 0.001,
    "prompt_token_limit": PROMPT_LIMIT,
    "response_token_limit": RESPONSE_LIMIT,
    "checkpoint_selection": "highest validation macro-F1; earliest tie",
    "device": str(device),
    "elapsed_seconds": time.time() - START
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
(OUT / "requirements.txt").write_text(
    "torch\nnumpy\npandas\nscikit-learn\nmatplotlib\nrequests\n"
)

print("\n========== FINAL RESULTS ==========")
print(pd.DataFrame(results).to_string(index=False))
print("\n========== DATASET ==========")
print(json.dumps(metadata, indent=2))
print("\n========== TRAINING TEXT STATISTICS ==========")
print(summary.to_string())

zip_path = "/content/midterm_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUT.iterdir():
        z.write(path, arcname=path.name)

print("\nDONE. Downloading results ZIP.")
from google.colab import files
files.download(zip_path)