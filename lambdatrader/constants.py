from enum import Enum


class PeriodEnum(Enum):
    M1 = 1 * 60
    M5 = 5 * 60
    M15 = 15 * 60
    H = 60 * 60
    H4 = 240 * 60
    D = 1440 * 60
    IRREGULAR = None

    def seconds(self) -> int:
        return self.value

    @classmethod
    def from_name(cls, name):
        mapping = {
            'M1': cls.M1,
            'M5': cls.M5,
            'M15': cls.M15,
            'H': cls.H,
            'H4': cls.H4,
            'D': cls.D,
            'IRREGULAR': cls.IRREGULAR
        }
        return mapping[name]


M1 = PeriodEnum.M1
M5 = PeriodEnum.M5
M15 = PeriodEnum.M15
H = PeriodEnum.H
H4 = PeriodEnum.H4
D = PeriodEnum.D
IRREGULAR = PeriodEnum.IRREGULAR

M1_SECONDS = M1.seconds()
M5_SECONDS = M5.seconds()
M15_SECONDS = M15.seconds()
H_SECONDS = H.seconds()
H4_SECONDS = H4.seconds()
D_SECONDS = D.seconds()


