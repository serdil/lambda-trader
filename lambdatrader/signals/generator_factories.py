from lambdatrader.constants import M5
from lambdatrader.signals.generators.constants import (
    LINREG__TP_STRATEGY_CLOSE_PRED_MULT, LINREG__TP_STRATEGY_MAX_PRED_MULT,
)
from lambdatrader.signals.generators.linreg import (
    LinRegSignalGeneratorSettings, LinRegSignalGenerator,
)
from lambdatrader.utilities.utils import seconds

LIN_REG_FIRST_CONF_SETTINGS = LinRegSignalGeneratorSettings(
                 num_candles=48,
                 candle_period=M5,
                 training_len=seconds(days=120),
                 train_val_ratio=0.6,
                 max_thr=0.05,
                 close_thr=0.04,
                 min_thr=-1.00,
                 max_rmse_thr=0.00,
                 close_rmse_thr=0.00,
                 max_rmse_mult=1.0,
                 close_rmse_mult=1.0,
                 use_rmse_for_close_thr=False,
                 use_rmse_for_max_thr=False,
                 tp_level=1.0,
                 tp_strategy=LINREG__TP_STRATEGY_MAX_PRED_MULT)

LIN_REG_SECOND_CONF_SETTINGS = LinRegSignalGeneratorSettings(
                 num_candles=48,
                 candle_period=M5,
                 training_len=seconds(days=500),
                 train_val_ratio=0.7,
                 max_thr=0.00,
                 close_thr=0.02,
                 min_thr=-1.00,
                 max_rmse_thr=0.03,
                 close_rmse_thr=0.02,
                 max_rmse_mult=1.0,
                 close_rmse_mult=1.0,
                 use_rmse_for_close_thr=True,
                 use_rmse_for_max_thr=False,
                 tp_level=1.0,
                 tp_strategy=LINREG__TP_STRATEGY_CLOSE_PRED_MULT)


class LinRegSignalGeneratorFactory:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.live = live
        self.silent = silent

    def get_first_conf_lin_reg_signal_generator(self):
        return self._create_with_settings(LIN_REG_FIRST_CONF_SETTINGS)

    def get_second_conf_lin_reg_signal_generator(self):
        return self._create_with_settings(LIN_REG_SECOND_CONF_SETTINGS)

    def _create_with_settings(self, settings):
        return LinRegSignalGenerator(self.market_info,
                                     live=self.live, silent=self.silent, settings=settings)
