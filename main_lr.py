import logging
import os
import random

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.metrics import weighted_mae
from src.models import RidgeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def extract_date_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["start_date"] = pd.to_datetime(df["start_date"], infer_datetime_format=True)
    df["month"] = df["start_date"].dt.month
    df["day"] = df["start_date"].dt.day
    # Normalize
    df["week"] = df["day"] // 7
    df.loc[df.query("week >= 4").index, "week"] = 3
    df["weekofyear"] = df["start_date"].dt.weekofyear
    return df


def extract_holiday(df: pd.DataFrame, month: int):
    month2holiday = {11: "Thanksgiving", 12: "Christmas"}
    holiday_name = month2holiday[month]
    df[holiday_name] = 0
    if month == 12:
        # 12月はIsHolidayになってる1week前にもpeekがある
        df.loc[df.query("IsHoliday==1 & month==@month").index, holiday_name] = 1
        df.loc[df.query("IsHoliday==1 & month==@month").index, holiday_name] = 1
    else:
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
    df = extract_holiday(df, month=11)
    df = extract_holiday(df, month=12)
    # One-Hot-Encodingとかが面倒なので、一旦categoricalは無視する
    # 一応target encodingをしているので、ある程度捉えられるはずs
    config.ridge.feature_cols = [
        col for col in config.ridge.feature_cols if col not in config.ridge.cat_cols
    ]
    # config.ridge.feature_cols = [
    #     col for col in config.ridge.feature_cols if ("avg" in col) or ("lag" in col)
    # ] + ["weekofyear"]
    df = calculate_weights(df)
    train_df = df[df[config.lgbm.label_col].notnull()]
    test_df = df[df[config.lgbm.label_col].isnull()]
    # 1年前の実績値がない場所は除く
    train_df = train_df.loc[df["lag1_store_dept_sales"].notnull()].reset_index(
        drop=True
    )
    # NULL埋め
    for col in config.ridge.feature_cols:
        if col != "lag1_store_dept_sales" or ("avg" not in col):
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(train_df[col].mean())
        else:
            # 実績値系は0で埋める
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)
    # 標準化
    std = StandardScaler()
    train_df.loc[:, config.ridge.feature_cols] = std.fit_transform(
        train_df[config.ridge.feature_cols]
    )
    test_df.loc[:, config.ridge.feature_cols] = std.transform(
        test_df[config.ridge.feature_cols]
    )
    model = RidgeModel(config.ridge)
    train_df, test_df = model.cv(train_df, test_df)
    model.save_model(
        config.store.model_path, suffix=f"_{config.feature.predict_months}_month"
    )
    # CVを計算
    metrics = []
    for n_fold in range(3):
        pred_col = f"{config.lgbm.label_col}_pred_fold{n_fold}"
        metrics.append(
            weighted_mae(train_df[train_df[pred_col].notnull()], pred_col=pred_col)
        )
    logger.info(f"CV: {np.mean(metrics)}")
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
