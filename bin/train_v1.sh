for predict_months in $(seq 1 10); do
    poetry run python main.py \
        feature=v1 \
        feature.predict_months=$predict_months \
        store.model_name=baseline 
done
