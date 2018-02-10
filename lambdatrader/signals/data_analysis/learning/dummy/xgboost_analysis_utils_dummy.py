import math

import numpy as np


def max_pred_key(a):
    return a[0][0]


def min_pred_key(a):
    return a[1][0]


def close_pred_key(a):
    return a[2][0]


def max_real_key(a):
    return a[0][1]


def min_real_key(a):
    return a[1][1]


def close_real_key(a):
    return a[2][1]


tp_level = 0.9


def tp_at_close_pred_profit(a):
    return close_pred_key(a) * tp_level \
        if one_if_tp_at_close_pred_hit(a) == 1 \
        else close_real_key(a)


def one_if_tp_at_close_pred_hit(a):
    return 1 \
        if close_pred_key(a) > 0 and max_real_key(a) >= close_pred_key(a) * tp_level \
        else 0


def get_tp_at_close_pred_thr_profit(thr):
    def tp_at_close_pred_thr_profit(a):
        return thr * tp_level \
            if get_one_if_tp_at_close_pred_thr_hit(thr)(a) == 1 \
            else close_real_key(a)
    return tp_at_close_pred_thr_profit


def get_one_if_tp_at_close_pred_thr_hit(thr):
    def one_if_tp_at_close_pred_thr_hit(a):
        return 1 \
            if thr > 0 and close_real_key(a) >= thr * tp_level \
            else 0
    return one_if_tp_at_close_pred_thr_hit


def tp_at_max_pred_profit(a):
    return max_pred_key(a) * tp_level \
        if one_if_tp_at_max_pred_hit(a) == 1 \
        else close_real_key(a)


def one_if_tp_at_max_pred_hit(a):
    return 1 \
        if max_pred_key(a) > 0 and max_real_key(a) >= max_pred_key(a) * tp_level \
        else 0


def get_tp_at_max_pred_thr_profit(thr):
    def tp_at_max_pred_thr_profit(a):
        return thr * tp_level \
            if get_one_if_tp_at_max_pred_thr_hit(thr)(a) == 1 \
            else close_real_key(a)
    return tp_at_max_pred_thr_profit


def get_one_if_tp_at_max_pred_thr_hit(thr):
    def one_if_tp_at_max_pred_thr_hit(a):
        return 1 \
            if thr > 0 and max_real_key(a) >= thr * tp_level \
            else 0
    return one_if_tp_at_max_pred_thr_hit


def one_if_max_pred_hit(a):
    return 1 if max_real_key(a) >= max_pred_key(a) else 0


def one_if_min_pred_hit(a):
    return 1 if min_real_key(a) >= min_pred_key(a) else 0


def one_if_close_pred_hit(a):
    return 1 if close_real_key(a) >= close_pred_key(a) else 0


def analyze_output(pred_real_max, pred_real_min, pred_real_close):
    pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))

    profits = [tp_at_close_pred_profit(a) for a in pred_real_max_min_close]

    pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))

    minimum_max_pred = max_pred_key(min(pred_real_max_min_close, key=max_pred_key))
    maximum_max_pred = max_pred_key(max(pred_real_max_min_close, key=max_pred_key))

    minimum_min_pred = min_pred_key(min(pred_real_max_min_close, key=min_pred_key))
    maximum_min_pred = min_pred_key(max(pred_real_max_min_close, key=min_pred_key))

    minimum_close_pred = close_pred_key(min(pred_real_max_min_close, key=close_pred_key))
    maximum_close_pred = close_pred_key(max(pred_real_max_min_close, key=close_pred_key))

    max_pred_step = 0.01
    max_pred_begin = max(math.floor(minimum_max_pred / max_pred_step) * max_pred_step, 0.00)
    max_pred_end = min(maximum_max_pred + max_pred_step, 0.10)

    min_pred_step = 0.01
    min_pred_begin = max(math.floor(minimum_min_pred / min_pred_step) * min_pred_step, -0.05)
    min_pred_end = min(maximum_min_pred + min_pred_step, 0.10)

    close_pred_step = 0.01
    close_pred_begin = max(math.floor(minimum_close_pred / close_pred_step)*close_pred_step, 0.00)
    close_pred_end = min(maximum_close_pred + close_pred_step, 0.10)

    for close_pred_thr in np.arange(close_pred_begin, close_pred_end, close_pred_step):
        for max_pred_thr in np.arange(max_pred_begin, max_pred_end, max_pred_step):
            for min_pred_thr in np.arange(min_pred_begin, min_pred_end, min_pred_step):
                filter_close = filter(lambda a: close_pred_key(a) >= close_pred_thr,
                                      pred_real_max_min_close)
                filter_max = filter(lambda a: max_pred_key(a) >= max_pred_thr, filter_close)
                filter_min = filter(lambda a: min_pred_key(a) >= min_pred_thr, filter_max)

                filtered = list(filter_min)

                if len(filtered) >= 10:
                    n_sig = len(filtered)

                    # TODO: implement and measure different TP and SL strategies

                    # tp strategies

                    close_pred_tp_count = sum(map(one_if_tp_at_close_pred_hit, filtered))
                    close_pred_tp_rate = close_pred_tp_count / n_sig
                    close_pred_total_profit = sum(map(tp_at_close_pred_profit, filtered))
                    close_pred_avg_profit = close_pred_total_profit / n_sig

                    max_pred_tp_count = sum(map(one_if_tp_at_max_pred_hit, filtered))
                    max_pred_tp_rate = max_pred_tp_count / n_sig
                    max_pred_total_profit = sum(map(tp_at_max_pred_profit, filtered))
                    max_pred_avg_profit = max_pred_total_profit / n_sig

                    one_if_close_pred_thr = get_one_if_tp_at_close_pred_thr_hit(close_pred_thr)
                    tp_at_close_pred_thr_profit = get_tp_at_close_pred_thr_profit(close_pred_thr)
                    close_pred_thr_tp_count = sum(map(one_if_close_pred_thr, filtered))
                    close_pred_thr_tp_rate = close_pred_thr_tp_count / n_sig
                    close_pred_thr_total_profit = sum(map(tp_at_close_pred_thr_profit, filtered))
                    close_pred_thr_avg_profit = close_pred_thr_total_profit / n_sig

                    one_if_max_pred_thr = get_one_if_tp_at_max_pred_thr_hit(max_pred_thr)
                    tp_at_max_pred_thr_profit = get_tp_at_max_pred_thr_profit(max_pred_thr)
                    max_pred_thr_tp_count = sum(map(one_if_max_pred_thr, filtered))
                    max_pred_thr_tp_rate = max_pred_thr_tp_count / n_sig
                    max_pred_thr_total_profit = sum(map(tp_at_max_pred_thr_profit, filtered))
                    max_pred_thr_avg_profit = max_pred_thr_total_profit / n_sig

                    # hit rates
                    max_pred_hit_count = sum(map(one_if_max_pred_hit, filtered))
                    max_pred_hit_rate = max_pred_hit_count / n_sig

                    min_pred_hit_count = sum(map(one_if_min_pred_hit, filtered))
                    min_pred_hit_rate = min_pred_hit_count / n_sig

                    close_pred_hit_count = sum(map(one_if_close_pred_hit, filtered))
                    close_pred_hit_rate = close_pred_hit_count / n_sig

                    # avg real max-min-close
                    real_max_sum = sum(map(max_real_key, filtered))
                    real_max_avg = real_max_sum / n_sig

                    real_min_sum = sum(map(min_real_key, filtered))
                    real_min_avg = real_min_sum / n_sig

                    real_close_sum = sum(map(close_real_key, filtered))
                    real_close_avg = real_close_sum / n_sig

                    print('cl_ma_mi {:<+5.2f} {:<+5.2f} {:<+5.2f} '
                          'n_sig {:<4} '
                          'cp_avgp {:<+8.5f} '
                          'mp_avgp {:<+8.5f} '
                          'cpt_avgp {:<+8.5f} '
                          'mpt_avgp {:<+8.5f} '
                          'cp_hr {:<+8.5f} '
                          'mp_hr {:<+8.5f} '
                          'max_avg {:<+8.5f} '
                          'close_avg {:<+8.5f} '
                          '                  '.format(close_pred_thr,
                                                      max_pred_thr,
                                                      min_pred_thr,
                                                      n_sig,
                                                      close_pred_avg_profit,
                                                      max_pred_avg_profit,
                                                      close_pred_thr_avg_profit,
                                                      max_pred_thr_avg_profit,
                                                      close_pred_hit_rate,
                                                      max_pred_hit_rate,
                                                      real_max_avg,
                                                      real_close_avg))

    # TODO: give some scores to model for some determined threshold levels

    print()
    for a in pred_real_max_min_close_profit[-10:]:
        print(a)
