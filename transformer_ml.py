"""
Transformer Next-Token GUI (Tkinter) — with Progress Tracking
============================================================
Goal: user types EXACTLY 4 words -> app shows the % likelihood of the next token.

What this script does:
- If a saved checkpoint exists -> load it instantly and run predictions.
- If no checkpoint exists -> trains a small causal Transformer LM on ./archive/*.txt,
  saves checkpoint, then the GUI becomes ready.

Files it will create:
- transformer_checkpoint.pt

Folder requirement:
- Put this script next to a folder named: archive/
  archive/ must contain your cleaned .txt files.

IMPORTANT NOTE (Windows CPU):
- Transformers can be slow on CPU. This file includes progress prints so you can see
  exactly where it's spending time.
"""

import os, re, math, random, time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tkinter as tk
from tkinter import ttk, messagebox


# -------------------------
# CONFIG
# -------------------------
ARCHIVE_DIR = "archive"
CHECKPOINT_PATH = "transformer_checkpoint.pt"

GUI_WORDS = 4

# Training context length
BLOCK_SIZE = 32

# Model size (CPU-friendly defaults)
EMBED_DIM = 128
N_HEADS = 4          # must divide EMBED_DIM evenly
N_LAYERS = 2
DROPOUT_P = 0.2

BATCH_SIZE = 16
MAX_EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-2
CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05

MIN_FREQ = 2
SEED = 42
DEVICE = "cpu"

VAL_FRAC = 0.1
PATIENCE = 3

# Repro
random.seed(SEED)
torch.manual_seed(SEED)

# CPU thread control (helps avoid thrashing)
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

# -------------------------
# PROGRESS LOGGER
# -------------------------
T0 = time.time()
def status(msg: str):
    dt = time.time() - T0
    print(f"[{dt:7.1f}s] {msg}", flush=True)


# -------------------------
# TOKENIZATION
# -------------------------
TOKEN_RE = re.compile(r"\w+|[^\w\s]")  # words OR punctuation

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


# -------------------------
# READ DATA
# -------------------------
def read_chunks(filepaths):
    """
    Each non-empty line becomes a mini-sequence.
    Avoids garbage transitions across file/paragraph boundaries.
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
# DATASET (Transformer blocks)
# -------------------------
class TransformerBlockDataset(Dataset):
    """
    Builds training samples for next-token prediction:
      x = [t0..t_{T-1}]
      y = [t1..t_T]
    where T = BLOCK_SIZE

    IMPORTANT:
    - To avoid building a *huge* dataset on CPU, we STRIDE by block_size.
      This reduces sample count ~block_size times.
    """
    def __init__(self, chunks_ids, block_size, bos_id, eos_id, pad_id):
        self.samples = []
        self.block_size = block_size

        for ids in chunks_ids:
            seq = [bos_id] + ids + [eos_id]
            if len(seq) < 2:
                continue

            # STRIDE by block_size (much faster than sliding every token)
            for start in range(0, len(seq) - 1, block_size):
                window = seq[start : start + (block_size + 1)]

                # pad to length block_size+1
                if len(window) < (block_size + 1):
                    window = window + [pad_id] * ((block_size + 1) - len(window))

                x = window[:-1]
                y = window[1:]
                self.samples.append((x, y))

                if start + (block_size + 1) >= len(seq):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -------------------------
# MODEL (Tiny causal Transformer LM)
# -------------------------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, n_heads, n_layers, dropout_p=0.1, tie_weights=True):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout_p)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying helps generalization and reduces params
        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} > block_size={self.block_size}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # [1, T]
        x = self.tok_emb(idx) + self.pos_emb(pos)                 # [B, T, E]
        x = self.drop(x)

        # causal mask blocks attention to future tokens
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        x = self.encoder(x, mask=causal_mask)  # [B, T, E]
        x = self.ln_f(x)
        logits = self.head(x)                  # [B, T, V]
        return logits


# -------------------------
# TRAINING HELPERS
# -------------------------
def split_chunks(chunks, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    chunks = chunks[:]
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * val_frac))
    return chunks[n_val:], chunks[:n_val]

def make_loader(chunks_ids, stoi, shuffle):
    ds = TransformerBlockDataset(
        chunks_ids,
        BLOCK_SIZE,
        bos_id=stoi[BOS],
        eos_id=stoi[EOS],
        pad_id=stoi[PAD],
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def eval_loss(model, loader, loss_fn, pad_id):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)         # [B, T, V]
            B, T, V = logits.shape

            logits = logits.reshape(B * T, V)
            y = y.reshape(B * T)

            loss = loss_fn(logits, y)

            nonpad = (y != pad_id).sum().item()
            total_loss += loss.item() * nonpad
            total_tokens += nonpad

    return total_loss / max(1, total_tokens)

def train_loop(model, train_loader, val_loader, stoi):
    pad_id = stoi[PAD]

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=LABEL_SMOOTHING)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        status(f"Starting epoch {epoch}...")
        model.train()
        total_loss, total_tokens = 0.0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)  # [B, T, V]
            B, T, V = logits.shape

            logits = logits.reshape(B * T, V)
            y = y.reshape(B * T)

            optim.zero_grad()
            loss = loss_fn(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optim.step()

            nonpad = (y != pad_id).sum().item()
            total_loss += loss.item() * nonpad
            total_tokens += nonpad

            if batch_idx == 0:
                status(f"  (epoch {epoch}) first batch done, loss={loss.item():.3f}")

        train_loss = total_loss / max(1, total_tokens)
        val_loss = eval_loss(model, val_loader, loss_fn, pad_id)

        train_ppl = math.exp(train_loss) if train_loss < 50 else float("inf")
        val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")

        print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} ppl={train_ppl:.2f} | val loss={val_loss:.4f} ppl={val_ppl:.2f}", flush=True)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            status("  ✓ New best val")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                status("  ✋ Early stopping (val stopped improving)")
                break


# -------------------------
# CHECKPOINT SAVE/LOAD
# -------------------------
def save_checkpoint(model, vocab, stoi, itos):
    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "stoi": stoi,
        "itos": itos,
        "config": {
            "block_size": BLOCK_SIZE,
            "embed_dim": EMBED_DIM,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout_p": DROPOUT_P,
            "min_freq": MIN_FREQ,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }
    torch.save(ckpt, CHECKPOINT_PATH)

def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    cfg = ckpt.get("config", {})

    block_size = int(cfg.get("block_size", BLOCK_SIZE))
    embed_dim = int(cfg.get("embed_dim", EMBED_DIM))
    n_heads = int(cfg.get("n_heads", N_HEADS))
    n_layers = int(cfg.get("n_layers", N_LAYERS))
    dropout_p = float(cfg.get("dropout_p", DROPOUT_P))

    vocab_size = len(ckpt["vocab"])

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout_p=dropout_p,
        tie_weights=True
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stoi, itos


# -------------------------
# TRAIN + BUILD
# -------------------------
def train_and_build():
    status("Starting train_and_build()")

    if not os.path.isdir(ARCHIVE_DIR):
        raise RuntimeError(f"Missing folder: {ARCHIVE_DIR}/")

    status("Scanning archive/ for .txt files...")
    files = [
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.lower().endswith(".txt")
    ]
    if not files:
        raise RuntimeError(f"No .txt files found in {ARCHIVE_DIR}/")
    status(f"Found {len(files)} text files")

    status("Reading chunks (lines)...")
    chunks = read_chunks(files)
    status(f"Loaded {len(chunks)} chunks")

    status("Flattening tokens + building vocab...")
    all_tokens = flatten(chunks)
    vocab, stoi, itos = build_vocab(all_tokens, min_freq=MIN_FREQ)
    status(f"Vocab size = {len(vocab)} | Total tokens = {len(all_tokens)}")

    status("Numericalizing chunks...")
    chunks_ids = [numericalize(toks, stoi) for toks in chunks]

    status("Splitting train/val...")
    train_chunks, val_chunks = split_chunks(chunks_ids, val_frac=VAL_FRAC, seed=SEED)
    status(f"Train chunks = {len(train_chunks)} | Val chunks = {len(val_chunks)}")

    status("Building DataLoaders...")
    train_loader = make_loader(train_chunks, stoi, shuffle=True)
    val_loader = make_loader(val_chunks, stoi, shuffle=False)
    status(f"DataLoaders ready | train batches/epoch ≈ {len(train_loader)}")

    status("Building model...")
    model = TinyTransformerLM(
        vocab_size=len(vocab),
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout_p=DROPOUT_P,
        tie_weights=True
    ).to(DEVICE)

    status("Starting training loop...")
    train_loop(model, train_loader, val_loader, stoi)

    status("Saving checkpoint...")
    save_checkpoint(model, vocab, stoi, itos)

    model.eval()
    status("Training done. Returning model.")
    return model, stoi, itos


# -------------------------
# PREDICT (GUI)
# -------------------------
@torch.no_grad()
def predict_topk(model, stoi, itos, four_words, topk=8):
    toks = tokenize(" ".join(four_words))
    ids = [stoi.get(t, stoi[UNK]) for t in toks]

    # pad to BLOCK_SIZE with BOS on the left
    bos_id = stoi[BOS]
    ids = ids[-BLOCK_SIZE:]
    if len(ids) < BLOCK_SIZE:
        ids = [bos_id] * (BLOCK_SIZE - len(ids)) + ids

    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    logits = model(x)                 # [1, T, V]
    last_logits = logits[0, -1, :]    # [V]
    probs = torch.softmax(last_logits, dim=0)

    k = min(topk, probs.numel())
    top_probs, top_ids = torch.topk(probs, k=k)

    out = [(itos[i.item()], 100 * p.item()) for p, i in zip(top_probs, top_ids)]
    return toks[-GUI_WORDS:], out


# -------------------------
# GUI
# -------------------------
class SimpleGUI(tk.Tk):
    def __init__(self, model, stoi, itos):
        super().__init__()
        self.title("Transformer LM (4 words → next token %)")

        self.model = model
        self.stoi = stoi
        self.itos = itos

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Type 4 words:").grid(row=0, column=0, sticky="w")
        self.entry = ttk.Entry(frm, width=60)
        self.entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.entry.insert(0, "frankly now i like")

        ttk.Button(frm, text="Predict", command=self.on_predict).grid(row=2, column=0, sticky="ew")

        self.out = tk.Text(frm, height=12, width=60)
        self.out.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(3, weight=1)

    def on_predict(self):
        self.out.delete("1.0", "end")

        raw = self.entry.get().strip()
        if not raw:
            messagebox.showwarning("Missing input", "Type 4 words first.")
            return

        words = raw.split()
        if len(words) != GUI_WORDS:
            messagebox.showwarning("Need exactly 4 words", f"You entered {len(words)} word(s).")
            return

        ctx_used, preds = predict_topk(self.model, self.stoi, self.itos, words, topk=8)
        self.out.insert("end", f"Tokens used: {ctx_used}\n\n")
        for tok, pct in preds:
            self.out.insert("end", f"{tok:>12}  {pct:6.2f}%\n")


# -------------------------
# MAIN
# -------------------------
def main():
    status("Program start")

    if os.path.isfile(CHECKPOINT_PATH):
        status("Checkpoint found. Loading...")
        try:
            model, stoi, itos = load_checkpoint()
            status("Checkpoint loaded. Launching GUI...")
        except Exception:
            status("Checkpoint mismatch or corrupt. Deleting and retraining...")
            os.remove(CHECKPOINT_PATH)
            model, stoi, itos = train_and_build()
    else:
        status("No checkpoint found. Training...")
        model, stoi, itos = train_and_build()

    status("Opening GUI window...")
    app = SimpleGUI(model, stoi, itos)
    app.mainloop()
    status("GUI closed. Exiting.")

if __name__ == "__main__":
    main()
