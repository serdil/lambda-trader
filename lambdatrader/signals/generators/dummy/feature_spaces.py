import random

from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.indicator_functions import PAT_REC_INDICATORS
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_LOW, OHLCV_CLOSE, OHLCV_HIGH, OHLCV_OPEN, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_features import (
    DFFeatureSet, SMASelfCloseDelta, CandlestickPattern, BBandsSelfCloseDelta, BBandsNowCloseDelta,
    RSIValue, MACDValue, SMANowCloseDelta, OHLCVNowCloseDelta, OHLCVSelfCloseDelta, OHLCVValue,
    BBandsBandWidth, CandlestickTipToTipSize, CandlestickBodySize, LinearFeatureCombination,
    PolynomialFeatureCombination,
)
from lambdatrader.signals.data_analysis.factories import FeatureSets


class ParameterRange:
    def __init__(self, parameter_type=None, start=None, end=None,
                 space=None, feature_set_sampler=None):
        self.use_space = space is not None
        self.use_fss = feature_set_sampler is not None
        if self.use_space:
            self.space = space
        elif self.use_fss:
            self.fss = feature_set_sampler
        elif parameter_type is None or start is None or end is None:
            raise ValueError('parameter_type, start and end should be non-none if space is None')
        elif type(start) != parameter_type or type(end) != parameter_type:
            raise ValueError('start and end should be of type {}'.format(parameter_type))
        self.parameter_type = parameter_type
        self.start = start
        self.end = end

    def sample(self):
        if self.use_space:
            return random.choice(self.space)
        elif self.use_fss:
            return self.fss.sample(size=1).features[0]
        elif self.parameter_type == int:
            return random.randint(self.start, self.end)
        elif self.parameter_type == float:
            return random.uniform(self.start, self.end)
        else:
            raise ValueError('unsupported parameter_type.')

    @classmethod
    def int_range(cls, start: int, end: int):
        return ParameterRange(int, start, end)

    @classmethod
    def float_range(cls, start: float, end: float):
        return ParameterRange(float, start, end)

    @classmethod
    def set(cls, elements):
        return ParameterRange(space=elements)

    @classmethod
    def features(cls, feature_set_sampler: FeatureSetSampler):
        return ParameterRange(feature_set_sampler=feature_set_sampler)


class BaseFeatureSampler:
    def sample(self, period=M5):
        raise NotImplementedError


class ParamRangeFeatureSampler(BaseFeatureSampler):
    def __init__(self, feature_class, parameter_ranges):
        self.feature_class = feature_class
        self.parameter_ranges = parameter_ranges

    def sample(self, period=M5):
        return self.feature_class(**self._get_kwargs(period=period))

    def _get_kwargs(self, period):
        kwargs = {'period': period}
        for param_name, param_range in self.parameter_ranges.items():
            kwargs[param_name] = param_range.sample()
        return kwargs


class LinearFeatureCombinationFeatureSampler(BaseFeatureSampler):
    def __init__(self, n_features_range: ParameterRange, coef_range: ParameterRange,
                 features: ParameterRange):
        self.n_features_range = n_features_range
        self.coef_range = coef_range
        self.features = features

    def sample(self, period=M5):
        n_features = self.n_features_range.sample()
        features = [self.features.sample() for _ in n_features]
        coef = [self.coef_range.sample() for _ in n_features]
        return LinearFeatureCombination(features=features, coef=coef, period=period)


class PolynomialFeatureCombinationFeatureSampler(BaseFeatureSampler):
    def __init__(self, n_features_range: ParameterRange, exp_range: ParameterRange,
                 features: ParameterRange):
        self.n_features_range = n_features_range
        self.exp_range = exp_range
        self.features = features

    def sample(self, period=M5):
        n_features = self.n_features_range.sample()
        features = [self.features.sample() for _ in n_features]
        exp = [self.exp_range.sample() for _ in n_features]
        return PolynomialFeatureCombination(features=features, exp=exp, period=period)


class FeatureSetSampler:
    def __init__(self, feature_samplers, periods=(M5, M15, H, H4, D)):
        self.feature_samplers = list(feature_samplers)
        self.periods = periods

    def sample(self, size=100):
        feature_names = set()
        features = []
        for _ in range(size):
            while True:
                period = random.choice(self.periods)
                feature_sampler = random.choice(self.feature_samplers)
                feature = feature_sampler.sample(period=period)
                if feature.name not in feature_names:
                    features.append(feature)
                    feature_names.add(feature.name)
                    break
        return FeatureSets.compose_remove_duplicates(DFFeatureSet(features=features))


feature_set_sampler_for_param_range = FeatureSetSampler(feature_samplers=[])


ohlc_now_close_delta_sampler = ParamRangeFeatureSampler(
    OHLCVNowCloseDelta,
    {
        'mode': ParameterRange.set([OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]),
        'offset': ParameterRange.int_range(0, 50),
    }
)

ohlc_self_close_delta_sampler = ParamRangeFeatureSampler(
    OHLCVSelfCloseDelta,
    {
        'mode': ParameterRange.set([OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]),
        'offset': ParameterRange.int_range(0, 50),
    }
)

volume_value_sampler = ParamRangeFeatureSampler(
    OHLCVValue,
    {
        'mode': ParameterRange.set([OHLCV_VOLUME]),
        'offset': ParameterRange.int_range(0, 50),
    }
)

sma_self_close_delta_sampler = ParamRangeFeatureSampler(
    SMASelfCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
    }
)

sma_now_close_delta_sampler = ParamRangeFeatureSampler(
    SMANowCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
    }
)

cs_pattern_sampler = ParamRangeFeatureSampler(
    CandlestickPattern,
    {
        'pattern_indicator': ParameterRange.set(PAT_REC_INDICATORS),
        'offset': ParameterRange.int_range(0, 10),
    }
)

bbands_self_close_delta_sampler = ParamRangeFeatureSampler(
    BBandsSelfCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'nbdevup': ParameterRange.int_range(1, 10),
        'nbdevdn': ParameterRange.int_range(1, 10),
        'matype': ParameterRange.int_range(0, 3),
        'offset': ParameterRange.int_range(0, 10),
        'output_col': ParameterRange.int_range(0, 2)
    }
)

bbands_now_close_delta_sampler = ParamRangeFeatureSampler(
    BBandsNowCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'nbdevup': ParameterRange.int_range(1, 10),
        'nbdevdn': ParameterRange.int_range(1, 10),
        'matype': ParameterRange.int_range(0, 3),
        'offset': ParameterRange.int_range(0, 10),
        'output_col': ParameterRange.int_range(0, 2)
    }
)

rsi_value_sampler = ParamRangeFeatureSampler(
    RSIValue,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
    }
)

macd_value_sampler = ParamRangeFeatureSampler(
    MACDValue,
    {
        'fastperiod': ParameterRange.int_range(2, 50),
        'slowperiod': ParameterRange.int_range(2, 50),
        'signalperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
        'output_col': ParameterRange.int_range(0, 2)
    }
)

bbands_band_width_sampler = ParamRangeFeatureSampler(
    BBandsBandWidth,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'nbdevup': ParameterRange.int_range(1, 10),
        'nbdevdn': ParameterRange.int_range(1, 10),
        'matype': ParameterRange.int_range(0, 3),
        'offset': ParameterRange.int_range(0, 10),
    }
)

cs_tip_to_tip_size_sampler = ParamRangeFeatureSampler(
    CandlestickTipToTipSize,
    {
        'offset': ParameterRange.int_range(0, 50),
    }
)

cs_body_size_sampler = ParamRangeFeatureSampler(
    CandlestickBodySize,
    {
        'offset': ParameterRange.int_range(0, 50),
    }
)

samplers_volume_value = [volume_value_sampler]

samplers_ohlc_self_close_delta = [ohlc_self_close_delta_sampler]

samplers_ohlcv = [ohlc_now_close_delta_sampler,
                  ohlc_self_close_delta_sampler,
                  volume_value_sampler]

samplers_all_old = [ohlc_now_close_delta_sampler,
                    ohlc_self_close_delta_sampler,
                    volume_value_sampler,
                    sma_self_close_delta_sampler,
                    sma_now_close_delta_sampler,
                    cs_pattern_sampler,
                    bbands_self_close_delta_sampler,
                    bbands_now_close_delta_sampler,
                    rsi_value_sampler,
                    macd_value_sampler]

samplers_all = [ohlc_now_close_delta_sampler,
                ohlc_self_close_delta_sampler,
                volume_value_sampler,
                sma_self_close_delta_sampler,
                sma_now_close_delta_sampler,
                cs_pattern_sampler,
                bbands_self_close_delta_sampler,
                bbands_now_close_delta_sampler,
                rsi_value_sampler,
                macd_value_sampler,
                bbands_band_width_sampler,
                cs_tip_to_tip_size_sampler,
                cs_body_size_sampler]

feature_set_sampler_for_param_range.feature_samplers = samplers_all


fs_sampler_volume_value = FeatureSetSampler(samplers_volume_value)
fs_sampler_ohlc_self_close_delta = FeatureSetSampler(samplers_ohlc_self_close_delta)
fs_sampler_ohlcv = FeatureSetSampler(samplers_ohlcv)
fs_sampler_all_old = FeatureSetSampler(samplers_all_old)
fs_sampler_all = FeatureSetSampler(samplers_all)
