"""
Simple Neural N-Gram GUI (Tkinter)
=================================
Goal: user types EXACTLY 4 words -> app shows the % likelihood of the next token.

What this script does:
- If a saved model checkpoint exists -> load it instantly and run predictions.
- If no checkpoint exists -> trains a small neural n-gram on ./archive/*.txt,
  saves checkpoint, then the GUI becomes ready.

Files it will create:
- ngram_checkpoint.pt

Folder requirement:
- Put this script next to a folder named: archive/
  archive/ must contain your cleaned .txt files.

Notes:
- CONTEXT_SIZE is fixed at 4 (because you want "4 words in").
- Prediction shows Top-K next tokens with percentages.
"""

import os
import re
import math
import random
from collections import Counter

# ---- CPU thread pool limits (set env vars BEFORE importing torch ideally) ----
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tkinter as tk
from tkinter import ttk, messagebox


# -------------------------
# CONFIG (keep simple)
# -------------------------
ARCHIVE_DIR = "archive"
CHECKPOINT_PATH = "ngram_checkpoint.pt"

CONTEXT_SIZE = 4          # fixed: you want 4 input words
EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT_P = 0.2

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0

MIN_FREQ = 2
SEED = 42
DEVICE = "cpu"

VAL_FRAC = 0.1
PATIENCE = 3
MAX_EPOCHS = 30
LABEL_SMOOTHING = 0.05


# DataLoader CPU knobs (you can set NUM_WORKERS=0 if Windows acts up)
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
PIN_MEMORY = False  # CPU only

# Repro
random.seed(SEED)
torch.manual_seed(SEED)

# CPU threading control
torch.set_num_threads(6)
torch.set_num_interop_threads(1)


# -------------------------
# TOKENIZATION
# -------------------------
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def read_all_tokens(filepaths):
    tokens = []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            tokens.extend(tokenize(f.read()))
    return tokens

def read_chunks(filepaths):
    """
    Better quality: treat each non-empty line as its own mini-sequence.
    This prevents the model from learning garbage transitions across file/paragraph boundaries.
    Returns: list of token-lists (chunks)
    """
    chunks = []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = tokenize(line)
                if toks:
                    chunks.append(toks)
    return chunks

def flatten(list_of_lists):
    out = []
    for x in list_of_lists:
        out.extend(x)
    return out

# -------------------------
# VOCAB
# -------------------------
PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"

def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = [PAD, UNK, BOS, EOS]
    for tok, c in counts.most_common():
        if c >= min_freq and tok not in vocab:
            vocab.append(tok)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos

def numericalize(tokens, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(t, unk_id) for t in tokens]


# -------------------------
# DATASET (memory efficient)
# -------------------------
class NGramChunkDataset(Dataset):
    """
    Builds (context, target) pairs from many independent chunks.
    Each chunk gets BOS... and EOS.
    """
    def __init__(self, chunks_token_ids, context_size, bos_id, eos_id):
        self.context_size = context_size
        self.examples = []

        for ids in chunks_token_ids:
            seq = [bos_id] * context_size + ids + [eos_id]
            for i in range(len(seq) - context_size):
                ctx = seq[i : i + context_size]
                tgt = seq[i + context_size]
                self.examples.append((ctx, tgt))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctx, tgt = self.examples[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# -------------------------
# MODEL
# -------------------------
class NeuralNGram(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size, dropout_p=0.2, tie_weights=True):
        super().__init__()
        self.context_size = context_size

        self.emb = nn.Embedding(vocab_size, embed_dim)

        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        # output projection
        self.fc2 = nn.Linear(hidden_dim, vocab_size, bias=False)

        # weight tying helps generalization
        if tie_weights:
            if hidden_dim != embed_dim:
                self.proj = nn.Linear(hidden_dim, embed_dim, bias=False)
                self.fc2 = nn.Linear(embed_dim, vocab_size, bias=False)
                self.fc2.weight = self.emb.weight
            else:
                self.proj = None
                self.fc2.weight = self.emb.weight
        else:
            self.proj = None

    def forward(self, context_ids):
        e = self.emb(context_ids)                  # [B, C, E]
        flat = e.reshape(e.size(0), -1)            # [B, C*E]
        h = self.fc1(flat)                         # [B, H]
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)

        if self.proj is not None:
            h = self.proj(h)

        logits = self.fc2(h)                       # [B, V]
        return logits

# -------------------------
# TRAINING
# -------------------------
def split_chunks(chunks, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    chunks = chunks[:]
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * val_frac))
    val = chunks[:n_val]
    train = chunks[n_val:]
    return train, val

def make_loader_from_chunks(chunks_ids, stoi, shuffle):
    bos_id = stoi[BOS]
    eos_id = stoi[EOS]
    ds = NGramChunkDataset(chunks_ids, CONTEXT_SIZE, bos_id, eos_id)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(ds, **loader_kwargs)

def eval_loss(model, loader, loss_fn):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for ctx, tgt in loader:
            ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
            logits = model(ctx)
            loss = loss_fn(logits, tgt)
            total_loss += loss.item() * tgt.size(0)
            total_n += tgt.size(0)
    return total_loss / max(1, total_n)

def train_and_save():
    if not os.path.isdir(ARCHIVE_DIR):
        raise RuntimeError(f"Missing folder: {ARCHIVE_DIR}/")

    files = [
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.lower().endswith(".txt")
    ]
    if not files:
        raise RuntimeError(f"No .txt files found in {ARCHIVE_DIR}/")

    # ---- better quality data: train within chunks (lines), not one giant stream ----
    chunks = read_chunks(files)                 # list[list[str]]
    all_tokens = flatten(chunks)                # list[str] for vocab building

    vocab, stoi, itos = build_vocab(all_tokens, min_freq=MIN_FREQ)

    # numericalize each chunk
    chunks_ids = [numericalize(toks, stoi) for toks in chunks]
    train_chunks, val_chunks = split_chunks(chunks_ids, val_frac=VAL_FRAC, seed=SEED)

    train_loader = make_loader_from_chunks(train_chunks, stoi, shuffle=True)
    val_loader = make_loader_from_chunks(val_chunks, stoi, shuffle=False)

    # ---- model ----
    model = NeuralNGram(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        context_size=CONTEXT_SIZE,
        dropout_p=DROPOUT_P,
        tie_weights=True
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)

            optimizer.zero_grad()
            logits = model(ctx)
            loss = loss_fn(logits, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            optimizer.step()

            total_loss += loss.item() * tgt.size(0)
            total_n += tgt.size(0)

        train_loss = total_loss / max(1, total_n)
        val_loss = eval_loss(model, val_loader, loss_fn)

        scheduler.step(val_loss)

        train_ppl = math.exp(train_loss) if train_loss < 50 else float("inf")
        val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")

        print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} ppl={train_ppl:.2f} | val loss={val_loss:.4f} ppl={val_ppl:.2f}")

        # save best checkpoint
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "context_size": CONTEXT_SIZE,
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout_p": DROPOUT_P,
                    "min_freq": MIN_FREQ,
                    "label_smoothing": LABEL_SMOOTHING,
                },
            }
            torch.save(ckpt, CHECKPOINT_PATH)
            print("  ✓ Saved best checkpoint")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("  ✋ Early stopping (val stopped improving)")
                break

    # load best before returning
    model, stoi, itos, context_size = load_checkpoint()
    return model, stoi, itos

def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    cfg = ckpt.get("config", {})

    # Respect checkpoint config (so changes in constants won't break loading)
    context_size = int(cfg.get("context_size", CONTEXT_SIZE))
    embed_dim = int(cfg.get("embed_dim", EMBED_DIM))
    hidden_dim = int(cfg.get("hidden_dim", HIDDEN_DIM))
    dropout_p = float(cfg.get("dropout_p", DROPOUT_P))

    vocab_size = len(ckpt["vocab"])

    model = NeuralNGram(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_size=context_size,
        dropout_p=dropout_p,
        tie_weights=True
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stoi, itos, context_size

# -------------------------
# PREDICT
# -------------------------
@torch.no_grad()
def predict_topk(model, stoi, itos, context_size, four_words, topk=8):
    # Tokenize exactly what they typed (so punctuation counts if present)
    toks = tokenize(" ".join(four_words))

    # We want exactly 4 tokens of context.
    # If they typed more/less due to punctuation, handle it robustly:
    ids = [stoi.get(t, stoi[UNK]) for t in toks][-context_size:]
    if len(ids) < context_size:
        ids = [stoi[BOS]] * (context_size - len(ids)) + ids

    ctx = torch.tensor([ids], dtype=torch.long).to(DEVICE)   # [1, C]
    logits = model(ctx).squeeze(0)                           # [V]
    probs = torch.softmax(logits, dim=0)                     # [V]

    k = min(topk, probs.numel())
    top_probs, top_ids = torch.topk(probs, k=k)

    out = []
    for p, i in zip(top_probs.tolist(), top_ids.tolist()):
        out.append((itos[i], 100.0 * p))
    return toks[-context_size:], out


# -------------------------
# SIMPLE GUI
# -------------------------
class SimpleGUI(tk.Tk):
    def __init__(self, model, stoi, itos, context_size):
        super().__init__()
        self.title("Neural N-Gram (4 words → next-word %)")

        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.context_size = context_size

        self._build()

    def _build(self):
        pad = 10
        frm = ttk.Frame(self, padding=pad)
        frm.grid(row=0, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(frm, text="Type 4 words (context):").grid(row=0, column=0, sticky="w")
        self.entry = ttk.Entry(frm, width=60)
        self.entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.entry.insert(0, "frankly now i like")

        btn = ttk.Button(frm, text="Predict next word %", command=self.on_predict)
        btn.grid(row=2, column=0, sticky="ew")

        ttk.Label(frm, text="Top predictions:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.out = tk.Text(frm, height=10, width=60)
        self.out.grid(row=4, column=0, sticky="nsew", pady=(4, 0))

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(4, weight=1)

    def on_predict(self):
        self.out.delete("1.0", "end")

        raw = self.entry.get().strip()
        if not raw:
            messagebox.showwarning("Missing input", "Type 4 words first.")
            return

        words = raw.split()
        if len(words) != 4:
            messagebox.showwarning("Need exactly 4 words", f"You entered {len(words)} word(s). Please enter exactly 4.")
            return

        ctx_used, preds = predict_topk(
            self.model, self.stoi, self.itos, self.context_size, words, topk=8
        )

        self.out.insert("end", f"Context tokens actually used (after tokenizer): {ctx_used}\n\n")
        for tok, pct in preds:
            self.out.insert("end", f"{tok:>12}  {pct:6.2f}%\n")


def main():
    # Load if possible, else train once then load
    if os.path.isfile(CHECKPOINT_PATH):
        model, stoi, itos, context_size = load_checkpoint()
    else:
        # Train (prints progress in console), then load from returned objects
        model, stoi, itos = train_and_save()
        context_size = CONTEXT_SIZE
        model.eval()

    app = SimpleGUI(model, stoi, itos, context_size)
    app.mainloop()


if __name__ == "__main__":
    main()
