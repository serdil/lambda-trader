class ConceptualTrade:
    def __init__(self, _id, start_date, end_date, profit):
        self.id = _id
        self.start_date = start_date
        self.end_date = end_date
        self.profit_amount = profit

    def __repr__(self):
        return 'Trade(start={}, end={}, profit={})'.format(
            self.start_date, self.end_date, self.profit_amount)
