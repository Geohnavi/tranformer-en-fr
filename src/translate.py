# src/translate.py
import argparse, json, torch
from transformer import TransformerENFR

def greedy_decode(model, src, max_len=50):
    device = next(model.parameters()).device
    src = src.unsqueeze(0).to(device)
    memory = model.encode(src, src == src_vocab["<pad>"])
    ys = torch.tensor([[tgt_vocab["<sos>"]]], device=device)
    for _ in range(max_len):
        out = model(src, ys, src == src_vocab["<pad>"], None)
        next_tok = out[0, -1].argmax().item()
        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
        if next_tok == tgt_vocab["<eos>"]: break
    return ys[0].tolist()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("sentence", nargs="+")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    src_vocab = {w: i for i, w in enumerate(ckpt["src_itos"])}
    tgt_vocab = {w: i for i, w in enumerate(ckpt["tgt_itos"])}
    inv_tgt = ckpt["tgt_itos"]

    model = TransformerENFR(src_vocab, tgt_vocab)
    model.load_state_dict(ckpt["model"]); model.eval()

    sent = " ".join(args.sentence)
    src_ids = [src_vocab.get(tok, src_vocab["<unk>"]) for tok in sent.split()]
    pred_ids = greedy_decode(model, torch.tensor(src_ids))
    words = [inv_tgt[i] for i in pred_ids[1:-1]]      # drop <sos> <eos>
    print(" ".join(words))
