from enum import Enum


class PeriodEnum(Enum):
    M5 = 5 * 60
    M15 = 15 * 60
    H = 60 * 60
    H4 = 240 * 60
    D = 1440 * 60

    def seconds(self) -> int:
        return self.value

M5 = PeriodEnum.M5
M15 = PeriodEnum.M15
H = PeriodEnum.H
H4 = PeriodEnum.H4
D = PeriodEnum.D

M5_SECONDS = M5.seconds()
M15_SECONDS = M15.seconds()
H_SECONDS = H.seconds()
H4_SECONDS = H4.seconds()
D_SECONDS = D.seconds()
