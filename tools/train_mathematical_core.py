"""Train and evaluate MCR-L's independent CPU syntactic comprehension core."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcrl.conllu import iter_conllu
from mcrl.syntax_model import train_models, evaluate_model

BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
FILES = {
    "train": "en_ewt-ud-train.conllu",
    "dev": "en_ewt-ud-dev.conllu",
    "test": "en_ewt-ud-test.conllu",
    "license": "LICENSE.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "MCR-L-mathematical-comprehension/1.0"})
    print(f"Downloading {url}")
    with urlopen(request, timeout=120) as response, target.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    print(f"  saved {target} ({target.stat().st_size:,} bytes)")


def ensure_corpus(directory: Path, force_download: bool = False) -> dict[str, Path]:
    files = {}
    for key, filename in FILES.items():
        target = directory / filename
        if force_download or not target.exists() or target.stat().st_size == 0:
            download(f"{BASE_URL}/{filename}", target)
        files[key] = target
    return files


def load_sentences(path: Path):
    sentences = list(iter_conllu(path))
    if not sentences:
        raise RuntimeError(f"No valid sentences were loaded from {path}")
    return sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "training" / "UD_English-EWT")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "models" / "mcrl_syntax_v1.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--download", action="store_true", help="download official train/dev/test CoNLL-U files and license")
    parser.add_argument("--force-download", action="store_true", help="replace local copies of all corpus files")
    parser.add_argument("--no-evaluate-test", action="store_true", help="skip final held-out test measurement")
    args = parser.parse_args()

    if args.epochs < 1 or args.epochs > 30:
        parser.error("--epochs must be between 1 and 30")
    try:
        files = ensure_corpus(args.data_dir, force_download=args.force_download)
        train = load_sentences(files["train"])
        dev = load_sentences(files["dev"])
        test = [] if args.no_evaluate_test else load_sentences(files["test"])

        print("\n=== Corpus ===")
        print(f"Train: {len(train):,} sentences / {sum(len(s) for s in train):,} tokens")
        print(f"Dev:   {len(dev):,} sentences / {sum(len(s) for s in dev):,} tokens")
        if test:
            print(f"Test:  {len(test):,} sentences / {sum(len(s) for s in test):,} tokens")
        print("Dataset: Universal Dependencies English Web Treebank (UD_English-EWT)")
        print("License: CC BY-SA 4.0 for the treebank annotations; retain attribution.")
        print("Note: source texts have their own rights; use the corpus under its published terms.")

        from mcrl.syntax_model import MathematicalSyntaxModel

        model, counts = train_models(
            train,
            epochs=args.epochs,
            progress=print,
        )
        model.metadata.update({
            "dataset": "Universal Dependencies English Web Treebank",
            "dataset_repository": "https://github.com/UniversalDependencies/UD_English-EWT",
            "dataset_release_reference": "master (download URL recorded in this training script)",
            "dataset_license": "CC BY-SA 4.0",
            "train_sha256": sha256(files["train"]),
            "dev_sha256": sha256(files["dev"]),
            "test_sha256": sha256(files["test"]),
            "train_sentences": len(train),
            "dev_sentences": len(dev),
            "epochs": args.epochs,
            "training_counts": counts,
            "architecture_note": (
                "Independent sparse averaged-perceptron POS tagger and arc-standard "
                "dependency transition parser. No LLM is used in this syntax stage."
            ),
        })

        print("\n=== Development evaluation ===")
        dev_metrics = evaluate_model(model, dev)
        for name, value in dev_metrics.items():
            print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value:,}")
        if test:
            print("\n=== Held-out test evaluation (do not tune against this set) ===")
            test_metrics = evaluate_model(model, test)
            for name, value in test_metrics.items():
                print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value:,}")
            model.metadata["held_out_test_metrics"] = test_metrics
        model.metadata["development_metrics"] = dev_metrics
        model.save(args.output)
        print(f"\nSaved mathematical syntax model: {args.output}")
        print("This model learns syntax only; it does not contain a complete world model or response policy.")
        print("Next run: py -3 -m mcrl --mathematical --interactive --debug")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
