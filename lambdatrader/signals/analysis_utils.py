from operator import itemgetter


def get_estimated_balances_list(trading_info):
    estimated_balances_dict = trading_info.estimated_balances
    estimated_balances_list = sorted(estimated_balances_dict.items(), key=itemgetter(0))
    return estimated_balances_list


def find_smaller_equal_date_index(estimated_balances_list, start_date):
    last_ind = 0
    for i, (date, balance) in enumerate(estimated_balances_list):
        if date > start_date:
            break
        last_ind = i
    return last_ind
