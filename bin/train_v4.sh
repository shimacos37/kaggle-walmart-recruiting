for predict_months in $(seq 1 10); do
    poetry run python main.py \
        feature=v4 \
        feature.predict_months=$predict_months \
        store.model_name=add_norm_feature 
done
