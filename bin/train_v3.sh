for predict_months in $(seq 1 10); do
    poetry run python main.py \
        feature=v3 \
        feature.predict_months=$predict_months \
        feature.version=3 \
        store.model_name=add_diff_feature
done
