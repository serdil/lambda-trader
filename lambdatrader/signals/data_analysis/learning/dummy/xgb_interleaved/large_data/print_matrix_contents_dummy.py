from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.factories import SplitDatasetDescriptors

sdd = SplitDatasetDescriptors

close_dataset = sdd.sdd_1_close_mini().training

# dataset = DFDataset.compute_from_descriptor(close_dataset, error_on_missing=False)

# print(dataset.value_df)

x, y = (DFDataset.compute_from_descriptor(descriptor=close_dataset, error_on_missing=False)
                 .add_feature_values()
                 .add_value_values(value_name=close_dataset.value_set.features[0].name)
                 .get())

print(x)
print(y)
