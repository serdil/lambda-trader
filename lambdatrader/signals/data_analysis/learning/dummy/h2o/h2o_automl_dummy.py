from h2o import H2OFrame, h2o
from h2o.automl import H2OAutoML

from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.signals.data_analysis.learning.dummy.dummy_utils_dummy import (
    get_dataset_info,
)
from lambdatrader.utilities.utils import seconds

h2o.init()

num_candles = 48

feature_set = FeatureSets.get_all_periods_last_five_ohlcv()
num_days = 500

ds_info = get_dataset_info(num_candles=num_candles, days=num_days, feature_set=feature_set)
feature_df, value_df, close_return_name, max_return_name, min_return_name = (ds_info.feature_df,
                                                                             ds_info.value_df,
                                                                             ds_info.close_return_name,
                                                                             ds_info.max_return_name,
                                                                             ds_info.min_return_name)

close_df = feature_df.join(value_df[close_return_name])

# print(close_df)

n_samples = len(close_df)

validation_ratio = 0.70
leaderboard_ratio = 0.85

validation_split_ind = int(n_samples * validation_ratio)
leaderboard_split_ind = int(n_samples * leaderboard_ratio)
gap = num_candles

close_train_df = close_df.iloc[0:validation_split_ind-gap]
close_val_df = close_df.iloc[validation_split_ind:leaderboard_split_ind-gap]
close_lead_df = close_df.iloc[leaderboard_split_ind:]

train_frame = H2OFrame(python_obj=close_train_df)
val_frame = H2OFrame(python_obj=close_val_df)
lead_frame = H2OFrame(python_obj=close_lead_df)

x = train_frame.columns
y = close_return_name
x.remove(y)

# run_seconds = 10
run_seconds = seconds(hours=8)

aml = H2OAutoML(max_runtime_secs=run_seconds, nfolds=0)
aml.train(x=x, y=y,
          training_frame=train_frame,
          validation_frame=val_frame,
          leaderboard_frame=lead_frame)

# View the AutoML Leaderboard
lb = aml.leaderboard

print(lb)

model_path = h2o.save_model(model=aml.leader, path="/tmp/nightmodel_h2o", force=True)

print('model saved:', model_path)
