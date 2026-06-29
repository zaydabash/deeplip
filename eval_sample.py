"""Quantitative + qualitative evaluation: run a checkpoint on clips and score WER/CER."""
import argparse
import glob
import os

from src.data import load_data
from src.dataset import build_vocab_lookup
from src.metrics import character_error_rate, word_error_rate
from src.predict import load_model, predict_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/weights_epoch_41.h5")
    parser.add_argument("--pattern", default="data/s1/*.mpg")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument(
        "--decoding", choices=["greedy", "beam"], default="greedy",
        help="CTC decoding strategy (default: greedy).",
    )
    args = parser.parse_args()

    char_to_num, num_to_char = build_vocab_lookup()
    model = load_model(args.weights)
    greedy = args.decoding == "greedy"

    clips = sorted(glob.glob(args.pattern))
    step = max(len(clips) // args.num, 1)
    clips = clips[::step][:args.num]

    wers, cers = [], []
    for path in clips:
        _, alignment = load_data(path, char_to_num)
        gt = "".join(c.decode() for c in num_to_char(alignment).numpy())
        pred = predict_clip(path, model, num_to_char, greedy=greedy)

        wer = word_error_rate(gt, pred)
        cer = character_error_rate(gt, pred)
        wers.append(wer)
        cers.append(cer)

        print(f"{os.path.basename(path):14} GT : {gt}")
        print(f"{'':14} OUT: {pred}  (WER {wer:.2f}, CER {cer:.2f})")

    print(f"\nAverage WER over {len(wers)} clips: {sum(wers) / len(wers):.3f}")
    print(f"Average CER over {len(cers)} clips: {sum(cers) / len(cers):.3f}")


if __name__ == "__main__":
    main()
