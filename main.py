import logging
import os
import random

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from src.metrics import weighted_mae
from src.models import LGBMModel
from src.utils import prepair_dir, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_date_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["start_date"] = pd.to_datetime(df["start_date"], infer_datetime_format=True)
    df["month"] = df["start_date"].dt.month
    df["day"] = df["start_date"].dt.day
    # Normalize
    df["week"] = df["day"] // 7
    df.loc[df.query("week >= 4").index, "week"] = 3
    df["weekofyear"] = df["start_date"].dt.weekofyear
    return df


def extract_holiday(df: pd.DataFrame, month: int) -> pd.DataFrame:
    month2holiday = {2: "SuperBowl", 11: "Thanksgiving", 12: "Christmas"}
    holiday_name = month2holiday[month]
    df[holiday_name] = 0
    df.loc[df.query("IsHoliday==1 & month==@month").index, holiday_name] = 1
    return df


def calculate_weights(df: pd.DataFrame) -> pd.DataFrame:
    df["weights"] = df["IsHoliday"].astype(int) * 4 + 1
    return df


@hydra.main(config_path="yamls/config.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    prepair_dir(config)
    set_seed(config.seed)
    df = pd.read_pickle(
        f"./input/train_{config.feature.predict_months}month_v{config.feature.version}.pkl"
    )
    df = extract_date_feature(df)
    df = extract_holiday(df, month=2)
    df = extract_holiday(df, month=11)
    df = extract_holiday(df, month=12)
    # Preprocess
    le_dict = {}
    for col in config.lgbm.cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    df = calculate_weights(df)
    train_df = df[df[config.lgbm.label_col].notnull()]
    test_df = df[df[config.lgbm.label_col].isnull()]
    model = LGBMModel(config.lgbm)
    train_df, test_df = model.cv(train_df, test_df)
    model.save_model(
        config.store.model_path, suffix=f"_{config.feature.predict_months}_month"
    )
    model.save_importance(
        config.store.result_path,
        suffix=f"_{config.feature.predict_months}_month",
    )
    # CVを計算
    metrics = []
    for n_fold in range(3):
        pred_col = f"{config.lgbm.label_col}_pred_fold{n_fold}"
        metrics.append(
            weighted_mae(train_df[train_df[pred_col].notnull()], pred_col=pred_col)
        )
    logger.info(f"CV: {np.mean(metrics)}")
    # Postprocess
    for col, le in le_dict.items():
        train_df[col] = le.inverse_transform(train_df[col])
        test_df[col] = le.inverse_transform(test_df[col])
    train_df.to_pickle(
        os.path.join(
            config.store.result_path,
            f"{config.feature.predict_months}month_train.pkl",
        ),
    )
    test_df[config.lgbm.label_col] = test_df[
        [f"{config.lgbm.label_col}_pred_fold{i}" for i in range(3)]
    ].mean(1)
    test_df.to_pickle(
        os.path.join(
            config.store.result_path,
            f"{config.feature.predict_months}month_test.pkl",
        ),
    )


if __name__ == "__main__":
    main()
