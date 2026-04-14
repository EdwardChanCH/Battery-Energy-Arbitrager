import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "clean")


def load_prices(split="train"):
    """Load electricity demand data as the price signal for the MDP.

    We use 'Adjusted demand' (MWh) as a proxy for electricity price,
    which is standard in energy arbitrage research when spot prices
    are unavailable. Higher demand correlates with higher prices.
    """
    if split == "train":
        df = pd.read_pickle(os.path.join(DATA_DIR, "train", "DFtrain.pkl"))
    elif split == "test":
        df = pd.read_pickle(os.path.join(DATA_DIR, "test", "DFtest.pkl"))
    elif split == "val":
        df = pd.read_pickle(os.path.join(DATA_DIR, "val", "DFval.pkl"))
    else:
        raise ValueError(f"Unknown split: {split}")

    prices = df["Adjusted demand"].values.astype(np.float32)

    # Forward-fill any zeros or NaNs
    mask = (prices == 0) | np.isnan(prices)
    if mask.any():
        s = pd.Series(prices)
        s[mask] = np.nan
        s = s.ffill().bfill()
        prices = s.values.astype(np.float32)

    return prices


if __name__ == "__main__":
    for split in ["train", "test"]:
        p = load_prices(split)
        print(f"{split}: {len(p)} hours, min={p.min():.0f}, max={p.max():.0f}, mean={p.mean():.0f}")
