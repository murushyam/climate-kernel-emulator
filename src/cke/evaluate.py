"""Summarise metrics.json files:  python -m cke.evaluate outputs/*/metrics.json"""
import json
import sys


def main(paths):
    print(f"{'run':30s} {'var':4s} {'NRMSE_s':>8s} {'NRMSE_g':>8s} {'NRMSE_t':>8s}")
    for p in paths:
        for v, m in json.load(open(p)).items():
            print(f"{p:30s} {v:4s} {m['nrmse_spatial']:8.3f} {m['nrmse_global']:8.3f} {m['nrmse_total']:8.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
