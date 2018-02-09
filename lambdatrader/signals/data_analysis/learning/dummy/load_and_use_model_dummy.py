from lambdatrader.shelve_cache import shelve_cache_get

model = shelve_cache_get('dummy_model_rfr_2592000')

print(model)
print(model.feature_importances_)
