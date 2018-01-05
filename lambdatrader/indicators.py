from enum import Enum

from talib.abstract import *

from lambdatrader.constants import M5
from lambdatrader.marketinfo import BaseMarketInfo


class IndicatorEnum(Enum):
    BBANDS = BBANDS                            #  Bollinger Bands
    DEMA = DEMA                                #  Double Exponential Moving Average
    EMA = EMA                                  #  Exponential Moving Average
    HT_TRENDLINE = HT_TRENDLINE                #  Hilbert Transform - Instantaneous Trendline
    KAMA = KAMA                                #  Kaufman Adaptive Moving Average
    MA = MA                                    #  Moving average
    MAMA = MAMA                                #  MESA Adaptive Moving Average
    MAVP = MAVP                                #  Moving average with variable period
    MIDPOINT = MIDPOINT                        #  MidPoint over period
    MIDPRICE = MIDPRICE                        #  Midpoint Price over period
    SAR = SAR                                  #  Parabolic SAR
    SAREXT = SAREXT                            #  Parabolic SAR - Extended
    SMA = SMA                                  #  Simple Moving Average
    T3 = T3                                    #  Triple Exponential Moving Average (T3)
    TEMA = TEMA                                #  Triple Exponential Moving Average
    TRIMA = TRIMA                              #  Triangular Moving Average
    WMA = WMA                                  #  Weighted Moving Average

    ADX = ADX                                  #  Average Directional Movement Index
    ADXR = ADXR                                #  Average Directional Movement Index Rating
    APO = APO                                  #  Absolute Price Oscillator
    AROON = AROON                              #  Aroon
    AROONOSC = AROONOSC                        #  Aroon Oscillator
    BOP = BOP                                  #  Balance Of Power
    CCI = CCI                                  #  Commodity Channel Index
    CMO = CMO                                  #  Chande Momentum Oscillator
    DX = DX                                    #  Directional Movement Index
    MACD = MACD                                #  Moving Average Convergence/Divergence
    MACDEXT = MACDEXT                          #  MACD with controllable MA type
    MACDFIX = MACDFIX                          #  Moving Average Convergence/Divergence Fix 12/26
    MFI = MFI                                  #  Money Flow Index
    MINUS_DI = MINUS_DI                        #  Minus Directional Indicator
    MINUS_DM = MINUS_DM                        #  Minus Directional Movement
    MOM = MOM                                  #  Momentum
    PLUS_DI = PLUS_DI                          #  Plus Directional Indicator
    PLUS_DM = PLUS_DM                          #  Plus Directional Movement
    PPO = PPO                                  #  Percentage Price Oscillator
    ROC = ROC                                  #  Rate of change : ((price/prevPrice)-1)*100
    ROCP = ROCP                                #  Rate of change Percentage: (price-prevPrice)/prevPrice
    ROCR = ROCR                                #  Rate of change ratio: (price/prevPrice)
    ROCR100 = ROCR100                          #  Rate of change ratio 100 scale: (price/prevPrice)*100
    RSI = RSI                                  #  Relative Strength Index
    STOCH = STOCH                              #  Stochastic
    STOCHF = STOCHF                            #  Stochastic Fast
    STOCHRSI = STOCHRSI                        #  Stochastic Relative Strength Index
    TRIX = TRIX                                #  1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    ULTOSC = ULTOSC                            #  Ultimate Oscillator
    WILLR = WILLR                              #  Williams' %R

    AD = AD                                    #  Chaikin A/D Line
    ADOSC = ADOSC                              #  Chaikin A/D Oscillator
    OBV = OBV                                  #  On Balance Volume

    HT_DCPERIOD = HT_DCPERIOD                  #  Hilbert Transform - Dominant Cycle Period
    HT_DCPHASE = HT_DCPHASE                    #  Hilbert Transform - Dominant Cycle Phase
    HT_PHASOR = HT_PHASOR                      #  Hilbert Transform - Phasor Components
    HT_SINE = HT_SINE                          #  Hilbert Transform - SineWave
    HT_TRENDMODE = HT_TRENDMODE                #  Hilbert Transform - Trend vs Cycle Mode

    AVGPRICE = AVGPRICE                        #  Average Price
    MEDPRICE = MEDPRICE                        #  Median Price
    TYPPRICE = TYPPRICE                        #  Typical Price
    WCLPRICE = WCLPRICE                        #  Weighted Close Price

    ATR = ATR                                  #  Average True Range
    NATR = NATR                                #  Normalized Average True Range
    TRANGE = TRANGE                            #  True Range

    CDL2CROWS = CDL2CROWS                      #  Two Crows
    CDL3BLACKCROWS = CDL3BLACKCROWS            #  Three Black Crows
    CDL3INSIDE = CDL3INSIDE                    #  Three Inside Up/Down
    CDL3LINESTRIKE = CDL3LINESTRIKE            #  Three-Line Strike
    CDL3OUTSIDE = CDL3OUTSIDE                  #  Three Outside Up/Down
    CDL3STARSINSOUTH = CDL3STARSINSOUTH        #  Three Stars In The South
    CDL3WHITESOLDIERS = CDL3WHITESOLDIERS      #  Three Advancing White Soldiers
    CDLABANDONEDBABY = CDLABANDONEDBABY        #  Abandoned Baby
    CDLADVANCEBLOCK = CDLADVANCEBLOCK          #  Advance Block
    CDLBELTHOLD = CDLBELTHOLD                  #  Belt-hold
    CDLBREAKAWAY = CDLBREAKAWAY                #  Breakaway
    CDLCLOSINGMARUBOZU = CDLCLOSINGMARUBOZU    #  Closing Marubozu
    CDLCONCEALBABYSWALL = CDLCONCEALBABYSWALL  #  Concealing Baby Swallow
    CDLCOUNTERATTACK = CDLCOUNTERATTACK        #  Counterattack
    CDLDARKCLOUDCOVER = CDLDARKCLOUDCOVER      #  Dark Cloud Cover
    CDLDOJI = CDLDOJI                          #  Doji
    CDLDOJISTAR = CDLDOJISTAR                  #  Doji Star
    CDLDRAGONFLYDOJI = CDLDRAGONFLYDOJI        #  Dragonfly Doji
    CDLENGULFING = CDLENGULFING                #  Engulfing Pattern
    CDLEVENINGDOJISTAR = CDLEVENINGDOJISTAR    #  Evening Doji Star
    CDLEVENINGSTAR = CDLEVENINGSTAR            #  Evening Star
    CDLGAPSIDESIDEWHITE = CDLGAPSIDESIDEWHITE  #  Up/Down-gap side-by-side white lines
    CDLGRAVESTONEDOJI = CDLGRAVESTONEDOJI      #  Gravestone Doji
    CDLHAMMER = CDLHAMMER                      #  Hammer
    CDLHANGINGMAN = CDLHANGINGMAN              #  Hanging Man
    CDLHARAMI = CDLHARAMI                      #  Harami Pattern
    CDLHARAMICROSS = CDLHARAMICROSS            #  Harami Cross Pattern
    CDLHIGHWAVE = CDLHIGHWAVE                  #  High-Wave Candle
    CDLHIKKAKE = CDLHIKKAKE                    #  Hikkake Pattern
    CDLHIKKAKEMOD = CDLHIKKAKEMOD              #  Modified Hikkake Pattern
    CDLHOMINGPIGEON = CDLHOMINGPIGEON          #  Homing Pigeon
    CDLIDENTICAL3CROWS = CDLIDENTICAL3CROWS    #  Identical Three Crows
    CDLINNECK = CDLINNECK                      #  In-Neck Pattern
    CDLINVERTEDHAMMER = CDLINVERTEDHAMMER      #  Inverted Hammer
    CDLKICKING = CDLKICKING                    #  Kicking
    CDLKICKINGBYLENGTH = CDLKICKINGBYLENGTH    #  Kicking - bull/bear determined by the longer marubozu
    CDLLADDERBOTTOM = CDLLADDERBOTTOM          #  Ladder Bottom
    CDLLONGLEGGEDDOJI = CDLLONGLEGGEDDOJI      #  Long Legged Doji
    CDLLONGLINE = CDLLONGLINE                  #  Long Line Candle
    CDLMARUBOZU = CDLMARUBOZU                  #  Marubozu
    CDLMATCHINGLOW = CDLMATCHINGLOW            #  Matching Low
    CDLMATHOLD = CDLMATHOLD                    #  Mat Hold
    CDLMORNINGDOJISTAR = CDLMORNINGDOJISTAR    #  Morning Doji Star
    CDLMORNINGSTAR = CDLMORNINGSTAR            #  Morning Star
    CDLONNECK = CDLONNECK                      #  On-Neck Pattern
    CDLPIERCING = CDLPIERCING                  #  Piercing Pattern
    CDLRICKSHAWMAN = CDLRICKSHAWMAN            #  Rickshaw Man
    CDLRISEFALL3METHODS = CDLRISEFALL3METHODS  #  Rising/Falling Three Methods
    CDLSEPARATINGLINES = CDLSEPARATINGLINES    #  Separating Lines
    CDLSHOOTINGSTAR = CDLSHOOTINGSTAR          #  Shooting Star
    CDLSHORTLINE = CDLSHORTLINE                #  Short Line Candle
    CDLSPINNINGTOP = CDLSPINNINGTOP            #  Spinning Top
    CDLSTALLEDPATTERN = CDLSTALLEDPATTERN      #  Stalled Pattern
    CDLSTICKSANDWICH = CDLSTICKSANDWICH        #  Stick Sandwich
    CDLTAKURI = CDLTAKURI                      #  Takuri (Dragonfly Doji with very long lower shadow)
    CDLTASUKIGAP = CDLTASUKIGAP                #  Tasuki Gap
    CDLTHRUSTING = CDLTHRUSTING                #  Thrusting Pattern
    CDLTRISTAR = CDLTRISTAR                    #  Tristar Pattern
    CDLUNIQUE3RIVER = CDLUNIQUE3RIVER          #  Unique 3 River
    CDLUPSIDEGAP2CROWS = CDLUPSIDEGAP2CROWS    #  Upside Gap Two Crows
    CDLXSIDEGAP3METHODS = CDLXSIDEGAP3METHODS  #  Upside/Downside Gap Three Methods

    def function(self):
        return self.value


class Indicators:

    def __init__(self, market_info: BaseMarketInfo):
        self.market_info = market_info

    def compute(self, pair, indicator: IndicatorEnum, args, ind=0, period=M5):
        indicator_function = indicator.function()
        indicator_input = self.get_input(pair=pair, ind=ind, period=period)
        range_results = indicator_function(indicator_input, *args)
        results_list = []
        for range_result in range_results:
            results_list.append(range_result[-1])
        return tuple(results_list)

    def get_input(self, pair, ind=0, period=M5, num_candles=100):
        input_candles = []
        for i in range(num_candles-1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, ind=ind, period=period)
            input_candles.append(candle)
        return {
            'open': self.candlesticks_open(input_candles),
            'high': self.candlesticks_high(input_candles),
            'low': self.candlesticks_low(input_candles),
            'close': self.candlesticks_close(input_candles),
            'volume': self.candlesticks_volume(input_candles)
        }

    @classmethod
    def candlesticks_open(cls, candlesticks):
        return cls.list_map(cls.candlestick_open, candlesticks)

    @classmethod
    def candlesticks_high(cls, candlesticks):
        return cls.list_map(cls.candlestick_high, candlesticks)

    @classmethod
    def candlesticks_low(cls, candlesticks):
        return cls.list_map(cls.candlestick_low, candlesticks)

    @classmethod
    def candlesticks_close(cls, candlesticks):
        return cls.list_map(cls.candlestick_close, candlesticks)

    @classmethod
    def candlesticks_volume(cls, candlesticks):
        return cls.list_map(cls.candlestick_volume, candlesticks)

    @staticmethod
    def list_map(func, items):
        return list(map(func, items))

    @staticmethod
    def candlestick_open(candlestick):
        return candlestick.open

    @staticmethod
    def candlestick_high(candlestick):
        return candlestick.high

    @staticmethod
    def candlestick_low(candlestick):
        return candlestick.low

    @staticmethod
    def candlestick_close(candlestick):
        return candlestick.close

    @staticmethod
    def candlestick_volume(candlestick):
        return candlestick.volume
