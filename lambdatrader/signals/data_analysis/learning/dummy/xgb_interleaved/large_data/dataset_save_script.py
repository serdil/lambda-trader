from lambdatrader.signals.data_analysis.df_datasets import XGBDMatrixDataset
from lambdatrader.signals.data_analysis.factories import SplitDatasetDescriptors

sdd = SplitDatasetDescriptors

save_libsvm = True
error_on_missing = False

# d1 = sdd.sdd_1_close_mini()
# d2 = sdd.sdd_1_max_mini()

# d1 = sdd.sdd_1_close()
# d2 = sdd.sdd_1_max()

d1 = sdd.sdd_1_more_data_close()
d2 = sdd.sdd_1_more_data_max()

print('d1 train:{} val:{} test:{}'.format(d1.training.hash, d1.validation.hash, d1.test.hash))
print('d2 train:{} val:{} test:{}'.format(d2.training.hash, d2.validation.hash, d2.test.hash))


if save_libsvm:
    XGBDMatrixDataset.save_libsvm(d1.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d1.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d1.test, error_on_missing=error_on_missing)

    XGBDMatrixDataset.save_libsvm(d2.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d2.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_libsvm(d2.test, error_on_missing=error_on_missing)
else:
    XGBDMatrixDataset.save_buffer(d1.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d1.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d1.test, error_on_missing=error_on_missing)

    XGBDMatrixDataset.save_buffer(d2.training, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d2.validation, error_on_missing=error_on_missing)
    XGBDMatrixDataset.save_buffer(d2.test, error_on_missing=error_on_missing)
