from enum import Enum


class PeriodEnum(Enum):
    M5 = 5 * 60
    M15 = 15 * 60
    H = 60 * 60
    H4 = 240 * 60
    D = 1440 * 60
