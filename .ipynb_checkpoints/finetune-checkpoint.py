from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

df = TimeSeriesDataFrame("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

predictor = TimeSeriesPredictor(prediction_length=48).fit(
    df,
    hyperparameters={
        "Chronos": {"model_path": "amazon/chronos-bolt-small"},
    },
)

predictions = predictor.predict(df)