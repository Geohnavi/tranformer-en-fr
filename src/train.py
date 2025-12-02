# src/train.py
import json, argparse, random, math
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformer import TransformerENFR

SOS, EOS, PAD, UNK = "<sos>", "<eos>", "<pad>", "<unk>"

def tokenize(text):                     # whitespace tokenizer
    return text.strip().split()

def build_vocab(path, min_freq=2):
    freq = {}
    for line in Path(path).open(encoding="utf8"):
        for tok in tokenize(line):
            freq[tok] = freq.get(tok, 0) + 1
    itos = [PAD, SOS, EOS, UNK] + [t for t, c in freq.items() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode(line, stoi):
    return [stoi.get(t, stoi[UNK]) for t in tokenize(line)]

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, tsv, src_stoi, tgt_stoi, max_len=50):
        self.pairs = []
        for ln in Path(tsv).open(encoding="utf8"):
            en, fr = ln.rstrip("\n").split("\t")[:2]
            s = encode(en, src_stoi)[:max_len-2]
            t = encode(fr, tgt_stoi)[:max_len-2]
            self.pairs.append(
                ([src_stoi[SOS]]+s+[src_stoi[EOS]],
                 [tgt_stoi[SOS]]+t+[tgt_stoi[EOS]]))
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate(batch):
    srcs, tgts = zip(*batch)
    src_len = max(len(s) for s in srcs)
    tgt_len = max(len(t) for t in tgts)
    pad = lambda seq, L, pad_id: seq + [pad_id]*(L-len(seq))
    src_pad = [pad(s, src_len, src_stoi[PAD]) for s in srcs]
    tgt_pad = [pad(t, tgt_len, tgt_stoi[PAD]) for t in tgts]
    return (torch.tensor(src_pad), torch.tensor(tgt_pad))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.txt")
    ap.add_argument("--valid", default="data/valid.txt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--save", default="checkpoints/enfr.pt")
    args = ap.parse_args()

    src_stoi, src_itos = build_vocab(args.train)
    tgt_stoi, tgt_itos = build_vocab(args.train.replace("train", "train"))  # same data
    Path("data/vocab_en.json").write_text(json.dumps(src_itos, ensure_ascii=False))
    Path("data/vocab_fr.json").write_text(json.dumps(tgt_itos, ensure_ascii=False))

    train_ds = PairDataset(args.train, src_stoi, tgt_stoi)
    valid_ds = PairDataset(args.valid, src_stoi, tgt_stoi)
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch,
                          shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerENFR(src_stoi, tgt_stoi,
                            d_model=args.d_model,
                            nhead=args.nhead,
                            layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss(ignore_index=tgt_stoi[PAD])

    best = math.inf
    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss = 0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            src_pad = (src == src_stoi[PAD])          # [batch, src_len]  True if PAD
            tgt_in  =  tgt[:, :-1]                    # decoder input (without last token)
            tgt_pad = (tgt_in == tgt_stoi[PAD])       # padding mask for decoder input
            out = model(src, tgt_in, src_pad, tgt_pad)
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()*src.size(0)
        print(f"epoch {epoch} train loss {tot_loss/len(train_ds):.4f}")

        # simple valid loss
        model.eval(); tot = 0
        with torch.no_grad():
            for src, tgt in valid_dl:
                src, tgt = src.to(device), tgt.to(device)
                src_pad = (src == src_stoi[PAD])          # [batch, src_len]  True if PAD
                tgt_in  =  tgt[:, :-1]                    # decoder input (without last token)
                tgt_pad = (tgt_in == tgt_stoi[PAD])       # padding mask for decoder input
                out = model(src, tgt_in, src_pad, tgt_pad)
                loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
                tot += loss.item()*src.size(0)
        val = tot/len(valid_ds)
        print(f"         valid loss {val:.4f}")
        if val < best:
            best = val
            torch.save({"model": model.state_dict(),
                        "src_itos": src_itos, "tgt_itos": tgt_itos}, args.save)
            print("         ✓ saved best model")
