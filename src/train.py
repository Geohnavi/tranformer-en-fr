# src/train.py  –  early-stopping + regularisation
import json, argparse, math, random
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformer import TransformerENFR

SOS, EOS, PAD, UNK = "<sos>", "<eos>", "<pad>", "<unk>"

# ─── helpers ─────────────────────────────────────────────────────────────
def tokenize(txt): return txt.strip().split()

def build_vocab(path, min_freq=2):
    freq = {}
    for line in Path(path).open(encoding="utf8"):
        for tok in tokenize(line):
            freq[tok] = freq.get(tok, 0) + 1
    itos = [PAD, SOS, EOS, UNK] + [t for t, c in freq.items() if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}, itos

def encode(line, stoi): return [stoi.get(t, stoi[UNK]) for t in tokenize(line)]

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, tsv, src_stoi, tgt_stoi, max_len=50):
        self.pairs = []
        for ln in Path(tsv).open(encoding="utf8"):
            en, fr = ln.rstrip("\n").split("\t")[:2]
            s = encode(en, src_stoi)[: max_len - 2]
            t = encode(fr, tgt_stoi)[: max_len - 2]
            self.pairs.append(
                ([src_stoi[SOS]] + s + [src_stoi[EOS]],
                 [tgt_stoi[SOS]] + t + [tgt_stoi[EOS]])
            )
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate(batch):
    srcs, tgts = zip(*batch)
    pad = lambda seq, L, pad_id: seq + [pad_id] * (L - len(seq))
    src_len, tgt_len = max(map(len, srcs)), max(map(len, tgts))
    src_pad = [pad(s, src_len, src_stoi[PAD]) for s in srcs]
    tgt_pad = [pad(t, tgt_len, tgt_stoi[PAD]) for t in tgts]
    return torch.tensor(src_pad), torch.tensor(tgt_pad)

# ─── CLI args ────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.txt")
ap.add_argument("--valid", default="data/valid.txt")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--d_model", type=int, default=512)
ap.add_argument("--nhead", type=int, default=8)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--patience", type=int, default=2,
                help="early-stop after N worse val epochs")
ap.add_argument("--save", default="checkpoints/enfr.pt")
args = ap.parse_args()

# ─── build vocab & data loaders ──────────────────────────────────────────
src_stoi, src_itos = build_vocab(args.train)
tgt_stoi, tgt_itos = build_vocab(args.train)
train_dl = DataLoader(PairDataset(args.train, src_stoi, tgt_stoi),
                      batch_size=args.batch, shuffle=True, collate_fn=collate)
valid_dl = DataLoader(PairDataset(args.valid, src_stoi, tgt_stoi),
                      batch_size=args.batch, shuffle=False, collate_fn=collate)

# ─── model / loss / opt ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TransformerENFR(src_stoi, tgt_stoi,
                         d_model=args.d_model,
                         nhead=args.nhead,
                         layers=args.layers,
                         dropout=args.dropout).to(device)
opt     = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
crit    = nn.CrossEntropyLoss(ignore_index=tgt_stoi[PAD], label_smoothing=0.1)

# ─── training loop with early-stopping ──────────────────────────────────
best_val = math.inf
bad_epochs = 0

for epoch in range(1, args.epochs + 1):
    # ---- train ---------------------------------------------------------
    model.train(); total_tr = 0
    for src, tgt in train_dl:
        src, tgt = src.to(device), tgt.to(device)
        src_pad = src.eq(src_stoi[PAD])
        tgt_in  = tgt[:, :-1]
        tgt_pad = tgt_in.eq(tgt_stoi[PAD])
        out = model(src, tgt_in, src_pad, tgt_pad)
        loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_tr += loss.item() * src.size(0)

    # ---- validate ------------------------------------------------------
    model.eval(); total_va = 0
    with torch.no_grad():
        for src, tgt in valid_dl:
            src, tgt = src.to(device), tgt.to(device)
            src_pad = src.eq(src_stoi[PAD])
            tgt_in  = tgt[:, :-1]
            tgt_pad = tgt_in.eq(tgt_stoi[PAD])
            out = model(src, tgt_in, src_pad, tgt_pad)
            vloss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            total_va += vloss.item() * src.size(0)

    train_loss = total_tr / len(train_dl.dataset)
    valid_loss = total_va / len(valid_dl.dataset)
    print(f"epoch {epoch:02d} train {train_loss:.4f}  valid {valid_loss:.4f}", end="")

    # ---- early-stop & checkpoint --------------------------------------
    if valid_loss < best_val - 0.01:            # significant improvement
        best_val = valid_loss; bad_epochs = 0
        torch.save({
    "model": model.state_dict(),
    "src_itos": src_itos,
    "tgt_itos": tgt_itos,
    "config": {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
        "dropout": args.dropout
        }
       }, args.save)
    else:
        bad_epochs += 1
        print()
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} bad epochs.")
            break
