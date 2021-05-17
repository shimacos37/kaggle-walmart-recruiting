for predict_months in $(seq 1 10); do
    poetry run python main.py \
        feature=v4 \
        feature.predict_months=$predict_months \
        store.model_name=add_norm_feature_lr001 \
        lgbm.params.learning_rate=0.01
done
