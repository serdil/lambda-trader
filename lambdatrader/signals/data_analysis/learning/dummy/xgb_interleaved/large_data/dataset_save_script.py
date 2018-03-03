from lambdatrader.signals.data_analysis.df_datasets import XGBDMatrixDataset
from lambdatrader.signals.data_analysis.factories import SplitDatasetDescriptors

sdd = SplitDatasetDescriptors

save_libsvm = True
error_on_missing = False

# d = sdd.sdd_1_close_mini()
# d = sdd.sdd_1_max_mini()

d = sdd.sdd_1_close()
# d = sdd.sdd_1_max()


if save_libsvm:
    XGBDMatrixDataset.save_libsvm(d.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d.test, error_on_missing=error_on_missing)
else:
    XGBDMatrixDataset.save_buffer(d.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d.test, error_on_missing=error_on_missing)
