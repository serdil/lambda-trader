import random

from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.indicator_functions import PAT_REC_INDICATORS
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_LOW, OHLCV_CLOSE, OHLCV_HIGH, OHLCV_OPEN,
)
from lambdatrader.signals.data_analysis.df_features import (
    DFFeatureSet, SMASelfCloseDelta, CandlestickPattern, BBandsSelfCloseDelta, BBandsNowCloseDelta,
    RSIValue, MACDValue, SMANowCloseDelta, OHLCVNowCloseDelta, OHLCVSelfCloseDelta, OHLCVValue,
)
from lambdatrader.signals.data_analysis.factories import FeatureSets


class ParameterRange:
    def __init__(self, parameter_type=None, start=None, end=None, space=None):
        self.use_space = space is not None
        if self.use_space:
            self.space = space
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
        if self.parameter_type == int:
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


class FeatureSampler:
    def __init__(self, feature_class, parameter_ranges):
        self.feature_class = feature_class
        self.parameter_ranges = parameter_ranges

    def sample(self):
        return self.feature_class(**self._get_kwargs())

    def _get_kwargs(self):
        kwargs = {}
        for param_name, param_range in self.parameter_ranges.items():
            kwargs[param_name] = param_range.sample()
        return kwargs


class FeatureSetSampler:
    def __init__(self, feature_samplers):
        self.feature_samplers = list(feature_samplers)

    def sample(self, size=100):
        features = []
        for _ in range(size):
            feature_sampler = random.choice(self.feature_samplers)
            features.append(feature_sampler.sample())
        return FeatureSets.compose_remove_duplicates(DFFeatureSet(features=features))


ohlc_now_close_delta_sampler = FeatureSampler(
    OHLCVNowCloseDelta,
    {
        'mode': ParameterRange.set([OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

ohlc_self_close_delta_sampler = FeatureSampler(
    OHLCVSelfCloseDelta,
    {
        'mode': ParameterRange.set([OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

volume_value_sampler = FeatureSampler(
    OHLCVValue,
    {
        'mode': ParameterRange.set([OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

sma_self_close_delta_sampler = FeatureSampler(
    SMASelfCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

sma_now_close_delta_sampler = FeatureSampler(
    SMANowCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

cs_pattern_sampler = FeatureSampler(
    CandlestickPattern,
    {
        'pattern_indicator': ParameterRange.set(PAT_REC_INDICATORS),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

bbands_self_close_delta_sampler = FeatureSampler(
    BBandsSelfCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'nbdevup': ParameterRange.int_range(1, 10),
        'nbdevdn': ParameterRange.int_range(1, 10),
        'matype': ParameterRange.int_range(0, 3),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

bbands_now_close_delta_sampler = FeatureSampler(
    BBandsNowCloseDelta,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'nbdevup': ParameterRange.int_range(1, 10),
        'nbdevdn': ParameterRange.int_range(1, 10),
        'matype': ParameterRange.int_range(0, 3),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

rsi_value_sampler = FeatureSampler(
    RSIValue,
    {
        'timeperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

macd_value_sampler = FeatureSampler(
    MACDValue,
    {
        'fastperiod': ParameterRange.int_range(2, 50),
        'slowperiod': ParameterRange.int_range(2, 50),
        'signalperiod': ParameterRange.int_range(2, 50),
        'offset': ParameterRange.int_range(0, 10),
        'period': ParameterRange.set([M5, M15, H, H4, D])
    }
)

all_samplers = [ohlc_now_close_delta_sampler,
                ohlc_self_close_delta_sampler,
                volume_value_sampler,
                sma_self_close_delta_sampler,
                sma_now_close_delta_sampler,
                cs_pattern_sampler,
                bbands_self_close_delta_sampler,
                bbands_now_close_delta_sampler,
                rsi_value_sampler,
                macd_value_sampler]

all_samplers_feature_set_sampler = FeatureSetSampler(all_samplers)
