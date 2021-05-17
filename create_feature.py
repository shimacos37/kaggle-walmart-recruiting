import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import hydra
import pandas as pd
from jinja2 import Template
from omegaconf import DictConfig


def load_query(config: DictConfig, predict_months: int) -> str:
    with open(f"./sql/walmart_feature_v{config.version}.sql") as f:
        template = Template(f.read())
    # 特徴量の接尾辞用
    groupby_names = [
        "_".join(groupby_col.lower().split(",")) for groupby_col in config.groupby_cols
    ]
    groupby_cols = zip(config.groupby_cols, groupby_names)
    aggregate_names = [aggregate_op.lower() for aggregate_op in config.aggregate_ops]
    aggregate_ops = zip(config.aggregate_ops, aggregate_names)
    # 訓練期間の前半部分は集計期間が短く、悪影響を及ぼしそうなので除く
    start_ts = datetime.fromisoformat(config.start_ts)
    start_ts += timedelta(days=28 * 3)
    # 予測期間のみを抜き出す (1ヶ月後予測なら1ヶ月後まで)
    end_ts = datetime.fromisoformat(config.end_ts)
    end_ts += timedelta(days=28 * predict_months)
    query = template.render(
        start_ts=start_ts.isoformat(),
        end_ts=end_ts.isoformat(),
        groupby_cols=list(groupby_cols),
        aggregate_ops=list(aggregate_ops),
        base_secs=3600 * 24 * 28 * predict_months,
        half_year_secs=3600 * 24 * 28 * 6,
        two_month_secs=3600 * 24 * 28 * 2,
        one_month_secs=3600 * 24 * 28 * 1,
    )
    return query


def save_dataframe(config: DictConfig, query: str, predict_months: int) -> None:
    df = pd.read_gbq(query, project_id=config.gcp.project_id, use_bqstorage_api=True)
    df.to_pickle(f"./input/train_{predict_months}month_v{config.feature.version}.pkl")


@hydra.main(config_path="./yamls/config.yaml")
def main(config: DictConfig):
    os.chdir(config.workdir)
    jobs = []
    thread_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    for predict_months in range(1, 11):
        query = load_query(config.feature, predict_months)
        jobs.append(
            thread_executor.submit(
                save_dataframe,
                config=config,
                query=query,
                predict_months=predict_months,
            )
        )
    for future in as_completed(jobs):
        jobs.remove(future)
        future.result()


if __name__ == "__main__":
    main()
