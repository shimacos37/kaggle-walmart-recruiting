import logging
import os
import pickle
from datetime import timedelta
from typing import Dict, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

plt.style.use("seaborn-whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_holdout(df: pd.DataFrame, n_fold: int):
    max_date = df["start_date"].max()
    split_date = max_date - timedelta(days=28 * (n_fold + 1))
    df.loc[df.query("start_date > @split_date").index, "fold"] = n_fold
    return df


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def _custom_objective(self, preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        weight = data.get_weight()
        grad = 2 * weight * (preds - labels)
        hess = 2 * weight
        return grad, hess

    def _custom_metric(self, preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        weight = data.get_weight()
        ae = np.abs(labels - preds) * weight
        wmae = ae.sum() / weight.sum()
        return "weighted_mae", wmae, False

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(3):
            train_df = calculate_holdout(train_df, n_fold)
            bst = self.fit(train_df, n_fold)
            valid_df = train_df.query("fold == @n_fold")
            train_df.loc[
                valid_df.index, f"{self.config.label_col}_pred_fold{n_fold}"
            ] = bst.predict(valid_df[self.config.feature_cols])
            test_df[f"{self.config.label_col}_pred_fold{n_fold}"] = bst.predict(
                test_df[self.config.feature_cols]
            )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance(importance_type="gain"))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
    ) -> lgb.Booster:

        X_train = train_df.query("fold!=@n_fold")[self.config.feature_cols]
        y_train = train_df.query("fold!=@n_fold")[self.config.label_col]

        X_valid = train_df.query("fold==@n_fold")[self.config.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        logger.info(f"{self.config.label_col} [Fold {n_fold}]")
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            weight=train_df.query("fold==@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        bst = lgb.train(
            dict(self.config.params),
            lgtrain,
            num_boost_round=100000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=self.config.early_stopping_rounds,
            categorical_feature=self.config.cat_cols,
            verbose_eval=self.config.verbose_eval,
            fobj=self._custom_objective,
            # feval=self._custom_metric,
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class RidgeModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for n_fold in range(3):
            train_df = calculate_holdout(train_df, n_fold)
            model = self.fit(train_df, n_fold)
            valid_df = train_df.query("fold == @n_fold")
            train_df.loc[
                valid_df.index, f"{self.config.label_col}_pred_fold{n_fold}"
            ] = model.predict(valid_df[self.config.feature_cols])
            test_df[f"{self.config.label_col}_pred_fold{n_fold}"] = model.predict(
                test_df[self.config.feature_cols]
            )
            self.store_model(model, n_fold)
        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
    ) -> Ridge:

        X_train = train_df.query("fold!=@n_fold")[self.config.feature_cols]
        y_train = train_df.query("fold!=@n_fold")[self.config.label_col]

        X_valid = train_df.query("fold==@n_fold")[self.config.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        logger.info(f"{self.config.label_col} [Fold {n_fold}]")
        best_mae = np.inf
        for alpha in np.linspace(0, 1, 10):
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            mae = mean_absolute_error(pred, y_valid)
            if mae <= best_mae:
                best_mae = mae
                best_model = model
        return best_model

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)
