for predict_months in $(seq 1 10); do
    poetry run python main_lr.py \
        feature=v2 \
        feature.predict_months=$predict_months \
        store.model_name=add_lag_feature_ridge 
done
