from lambdatrader.signals.generators.dummy.feature_spaces import fs_sampler_all

sampler = fs_sampler_all

feature_set = sampler.sample(size=1000)

for f in feature_set.sample():
    print(f.name)
