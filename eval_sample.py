"""Quick qualitative evaluation: run a checkpoint on a few clips vs ground truth."""
import argparse
import glob
import os

from src.data import load_data
from src.dataset import build_vocab_lookup
from src.predict import load_model, predict_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/weights_epoch_41.h5")
    parser.add_argument("--pattern", default="data/s1/*.mpg")
    parser.add_argument("--num", type=int, default=6)
    args = parser.parse_args()

    char_to_num, num_to_char = build_vocab_lookup()
    model = load_model(args.weights)

    clips = sorted(glob.glob(args.pattern))
    step = max(len(clips) // args.num, 1)
    clips = clips[::step][:args.num]

    total_words = 0
    correct_words = 0
    for path in clips:
        _, alignment = load_data(path, char_to_num)
        gt = "".join(c.decode() for c in num_to_char(alignment).numpy())
        pred = predict_clip(path, model, num_to_char)

        gt_w, pred_w = gt.split(), pred.split()
        total_words += len(gt_w)
        correct_words += sum(1 for g, h in zip(gt_w, pred_w) if g == h)

        print(f"{os.path.basename(path):14} GT : {gt}")
        print(f"{'':14} OUT: {pred}")

    print(f"\nrough word match (positional): {correct_words}/{total_words}")


if __name__ == "__main__":
    main()
