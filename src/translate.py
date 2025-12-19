# src/translate.py
import argparse, random, torch
from transformer import TransformerENFR

def make_masks(src, tgt, pad_id):
    src_pad = src.eq(pad_id)
    tgt_pad = tgt.eq(pad_id)
    T = tgt.size(1)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), 1)
    return src_pad, tgt_pad, causal

def clean_logits(logits, generated, ngram=3, max_freq=2):
    # ① block n-gram repeats
    if generated.size(1) >= ngram:
        prefix = tuple(generated[0, -ngram + 1 :].tolist())
        banned = {
            generated[0, i + ngram - 1].item()
            for i in range(generated.size(1) - ngram + 1)
            if tuple(generated[0, i : i + ngram - 1].tolist()) == prefix
        }
        if banned:
            logits[list(banned)] = -1e9

    # ② block tokens seen too often
    counts = torch.bincount(generated.view(-1))
    if counts.numel():
        over = (counts > max_freq).nonzero().flatten().tolist()
        if over:
            logits[over] = -1e9
    return logits

def greedy_decode(model, src, k=10, p=0.95, tau=0.9, max_len=50):
    device = next(model.parameters()).device
    src = src.unsqueeze(0).to(device)
    src_pad = src.eq(src_vocab["<pad>"])
    memory = model.encode(src, src_pad)

    gen = torch.tensor([[tgt_vocab["<sos>"]]], device=device)
    for _ in range(max_len):
        src_pad, tgt_pad, _ = make_masks(src, gen, tgt_vocab["<pad>"])
        logits = model(src, gen, src_pad, tgt_pad)[0, -1]

        logits[tgt_vocab["<unk>"]] = -1e9
        logits = clean_logits(logits.clone(), gen)
        logits /= tau                                           # temperature

        # top-k
        topv, topi = torch.topk(logits, k)
        logits = torch.full_like(logits, -1e9)
        logits[topi] = topv

        # nucleus
        probs = torch.softmax(logits, -1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, 0)
        probs[sorted_i[cum > p]] = 0

        s = probs.sum()
        if not torch.isfinite(s) or s.item() == 0.0:
            next_id = logits.argmax().item()                    # deterministic fallback
        else:
            probs /= s
            next_id = random.choices(range(len(probs)), weights=probs.tolist())[0]

        gen = torch.cat([gen, torch.tensor([[next_id]], device=device)], 1)
        if next_id == tgt_vocab["<eos>"]:
            break
    return gen[0].tolist()

def beam_search(model, src, beam=4, alpha=0.7, max_len=50):
    device = next(model.parameters()).device
    src = src.unsqueeze(0).to(device)
    src_pad = src.eq(src_vocab["<pad>"])
    memory = model.encode(src, src_pad)

    beams = [([tgt_vocab["<sos>"]], 0.0)]                        # (seq, score)
    for _ in range(max_len):
        cand = []
        for seq, score in beams:
            if seq[-1] == tgt_vocab["<eos>"]:
                cand.append((seq, score))
                continue
            tgt = torch.tensor([seq], device=device)
            src_pad_, tgt_pad_, _ = make_masks(src, tgt, tgt_vocab["<pad>"])
            logits = model(src, tgt, src_pad_, tgt_pad_)[0, -1]
            logits[tgt_vocab["<unk>"]] = -1e9
            logits = clean_logits(logits.clone(), tgt)
            log_p = logits.log_softmax(-1)
            for tok in torch.topk(log_p, beam).indices.tolist():
                new_seq = seq + [tok]
                lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha)
                cand.append((new_seq, score + log_p[tok].item() / lp))
        beams = sorted(cand, key=lambda x: x[1], reverse=True)[: beam]
        if all(s[-1] == tgt_vocab["<eos>"] for s, _ in beams):
            break
    return beams[0][0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model checkpoint (.pt)")
    ap.add_argument("--beam", type=int, default=1, help="beam width (1 = greedy)")
    ap.add_argument("--alpha", type=float, default=0.7, help="length-penalty for beam")
    ap.add_argument("--k", type=int, default=10, help="top-k for greedy")
    ap.add_argument("--p", type=float, default=0.95, help="nucleus p for greedy")
    ap.add_argument("--tau", type=float, default=0.9, help="temperature for greedy")
    ap.add_argument("sentence", nargs="+")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    src_vocab = {w: i for i, w in enumerate(ckpt["src_itos"])}
    tgt_vocab = {w: i for i, w in enumerate(ckpt["tgt_itos"])}

    inv_tgt = ckpt["tgt_itos"]

    model = TransformerENFR(
    src_vocab,
    tgt_vocab,
    d_model=512,
    nhead=8,
    layers=6,
    dropout=0.2
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    src_ids = [src_vocab.get(tok, src_vocab["<unk>"]) for tok in " ".join(args.sentence).split()]
    src_tensor = torch.tensor(src_ids)

    if args.beam == 1:
        ids = greedy_decode(model, src_tensor, k=args.k, p=args.p, tau=args.tau)
    else:
        ids = beam_search(model, src_tensor, beam=args.beam, alpha=args.alpha)

    print(" ".join(inv_tgt[i] for i in ids[1:-1]))  # drop <sos>/<eos>
