import random, pathlib

def main():
    src = pathlib.Path("data/eng-fra.tsv")
    pairs = []
    with src.open(encoding="utf8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:                 # make sure we have at least EN + FR
                en, fr = parts[0], parts[1]     # grab just the first two columns
                pairs.append((en, fr))

    random.shuffle(pairs)

    n = len(pairs); tr, va = int(n*0.9), int(n*0.95)
    splits = [("train.txt", pairs[:tr]),
              ("valid.txt", pairs[tr:va]),
              ("test.txt",  pairs[va:])]

    for name, chunk in splits:
        out = pathlib.Path("data") / name
        with out.open("w", encoding="utf8") as f:
            for en, fr in chunk:
                print(en, fr, sep="\t", file=f)

if __name__ == "__main__":
    main()