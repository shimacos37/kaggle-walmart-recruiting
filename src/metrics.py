import numpy as np
import pandas as pd


def weighted_mae(
    df: pd.DataFrame,
    label_col: str = "Weekly_Sales",
    pred_col: str = "Weekly_Sales_pred",
) -> float:
    wmae = np.abs(df[label_col] - df[pred_col]) * df["weights"]
    wmae = wmae.sum() / df["weights"].sum()
    return wmae
