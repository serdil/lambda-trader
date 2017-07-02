class Trade:
    def __init__(self, id, start_date, end_date, profit):
        self.id = id
        self.start_date = start_date
        self.end_date = end_date
        self.profit_amount = profit

    def __repr__(self):
        return 'Trade(start={}, end={}, profit={})'.format(
            self.start_date, self.end_date, self.profit_amount)
