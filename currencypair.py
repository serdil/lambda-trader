class CurrencyPair:
    def __init__(self, first_currency, second_currency):
        self.first = first_currency
        self.second = second_currency

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.first == other.first and self.second == other.second
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.first, self.second))
