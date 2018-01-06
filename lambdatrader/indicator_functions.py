from enum import Enum
from math import sqrt

from talib.abstract import (
    BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, SAREXT, SMA, T3,
    TEMA, TRIMA, WMA, ADX, ADXR, APO, AROON, AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX,
    MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH,
    STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR, AD, ADOSC, OBV, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR,
    HT_SINE, HT_TRENDMODE, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, ATR, NATR, TRANGE, CDL2CROWS,
    CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS,
    CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU,
    CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR,
    CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE,
    CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE,
    CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING,
    CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU,
    CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING,
    CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE,
    CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP, CDLTHRUSTING,
    CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS,
)


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

    CSTHMA = HMA

    def function(self):
        return self.value


def HMA(inputs, period):
    wma_func = IndicatorEnum.WMA.function()
    wma = wma_func(inputs, period)
    wma_half_period = wma_func(inputs, period // 2)
    hma = wma_func(2 * wma_half_period - wma, sqrt(period))
    return hma
