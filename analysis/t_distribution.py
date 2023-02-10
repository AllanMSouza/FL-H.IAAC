import math
import scipy
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

def t_distribution_test(x, confidence=0.95):
    n = len(x)
    decimals = 1
    mean = round(st.mean(x), decimals)
    liberty_graus = n
    s = st.stdev(x)
    alfa = 1 - confidence
    column = 1 - alfa / 2
    t_value = T_Distribution().find_t_distribution(column, liberty_graus)
    average_variation = round((t_value * (s / math.pow(n, 1 / 2))), decimals)
    average_variation = str(average_variation)
    while len(average_variation) < decimals + 3:
        average_variation = "0" + average_variation

    mean = str(mean)
    while len(mean) < decimals + 3:
        mean = "0" + mean

    ic = st.t.interval(alpha=0.95, df=len(x) - 1, loc=np.mean(x), scale=st.sem(x))
    l = round(ic[0], decimals)
    r = round(ic[1], decimals)

    return str(mean) + u"\u00B1" + average_variation

class T_Distribution:

    def __init__(self):

        self.q_60 = [0.325,
                    0.289,
                    0.277,
                    0.271,
                    0.267,
                    0.265,
                    0.263,
                    0.262,
                    0.261,
                    0.260,
                    0.260,
                    0.259,
                    0.259,
                    0.258,
                    0.258,
                    0.258,
                    0.257,
                    0.257,
                    0.257,
                    0.257,
                    0.257,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.254,
                    0.254,
                    0.254]

        self.q_70 = [0.727,
                    0.617,
                    0.584,
                    0.569,
                    0.559,
                    0.553,
                    0.549,
                    0.546,
                    0.543,
                    0.542,
                    0.540,
                    0.539,
                    0.538,
                    0.537,
                    0.536,
                    0.535,
                    0.534,
                    0.534,
                    0.533,
                    0.533,
                    0.532,
                    0.532,
                    0.532,
                    0.531,
                    0.531,
                    0.531,
                    0.531,
                    0.530,
                    0.530,
                    0.530,
                    0.527,
                    0.526,
                    0.526]

        self.q_80 = [1.377,
                    1.061,
                    0.978,
                    0.941,
                    0.920,
                    0.906,
                    0.896,
                    0.889,
                    0.883,
                    0.879,
                    0.876,
                    0.873,
                    0.870,
                    0.868,
                    0.866,
                    0.865,
                    0.863,
                    0.862,
                    0.861,
                    0.860,
                    0.859,
                    0.858,
                    0.858,
                    0.857,
                    0.856,
                    0.856,
                    0.855,
                    0.855,
                    0.854,
                    0.854,
                    0.848,
                    0.846,
                    0.845]

        self.q_90 = [3.078,
                        1.886,
                        1.638,
                        1.533,
                        1.476,
                        1.440,
                        1.415,
                        1.397,
                        1.383,
                        1.372,
                        1.363,
                        1.356,
                        1.350,
                        1.345,
                        1.341,
                        1.337,
                        1.333,
                        1.330,
                        1.328,
                        1.325,
                        1.323,
                        1.321,
                        1.319,
                        1.318,
                        1.316,
                        1.315,
                        1.314,
                        1.313,
                        1.311,
                        1.310,
                        1.296,
                        1.291,
                        1.289]

        self.q_95 = [6.314,
                    2.920,
                    2.353,
                    2.132,
                    2.015,
                    1.943,
                    1.895,
                    1.860,
                    1.833,
                    1.812,
                    1.796,
                    1.782,
                    1.771,
                    1.761,
                    1.753,
                    1.746,
                    1.740,
                    1.734,
                    1.729,
                    1.725,
                    1.721,
                    1.717,
                    1.714,
                    1.711,
                    1.708,
                    1.706,
                    1.703,
                    1.701,
                    1.699,
                    1.697,
                    1.671,
                    1.662,
                    1.658]

        self.q_975 = [12.706,
                    4.303,
                    3.182,
                    2.776,
                    2.571,
                    2.447,
                    2.365,
                    2.306,
                    2.262,
                    2.228,
                    2.201,
                    2.179,
                    2.160,
                    2.145,
                    2.131,
                    2.120,
                    2.110,
                    2.101,
                    2.093,
                    2.086,
                    2.080,
                    2.074,
                    2.069,
                    2.064,
                    2.060,
                    2.056,
                    2.052,
                    2.048,
                    2.045,
                    2.042,
                    2.000,
                    1.987,
                    1.980]

        self.q_995 = [63.657,
                    9.925,
                    5.841,
                    4.604,
                    4.032,
                    3.707,
                    3.499,
                    3.355,
                    3.250,
                    3.169,
                    3.106,
                    3.055,
                    3.012,
                    2.977,
                    2.947,
                    2.921,
                    2.898,
                    2.878,
                    2.861,
                    2.845,
                    2.831,
                    2.819,
                    2.807,
                    2.797,
                    2.787,
                    2.779,
                    2.771,
                    2.763,
                    2.756,
                    2.750,
                    2.660,
                    2.632,
                    2.617]

        self.q_9995 = [636.619,
                    31.599,
                    12.924,
                    8.610,
                    6.869,
                    5.959,
                    5.408,
                    5.041,
                    4.781,
                    4.587,
                    4.437,
                    4.318,
                    4.221,
                    4.140,
                    4.073,
                    4.015,
                    3.965,
                    3.922,
                    3.883,
                    3.850,
                    3.819,
                    3.792,
                    3.768,
                    3.745,
                    3.725,
                    3.707,
                    3.690,
                    3.674,
                    3.659,
                    3.646,
                    3.460,
                    3.402,
                    3.373]

        self.t_table = pd.DataFrame({'60': self.q_60, '70': self.q_70,
                                     '80': self.q_80, '90': self.q_90,
                                     '95': self.q_95, '975': self.q_975,
                                     '995': self.q_995, '9995': self.q_9995})

    def find_t_distribution(self, column, v):

        v-=1

        if column == 0.6:
            quantile = self.t_table['60'].iloc[v]
        elif column == 0.7:
            quantile = self.t_table['70'].iloc[v]
        elif column == 0.8:
            quantile = self.t_table['80'].iloc[v]
        elif column == 0.9:
            quantile = self.t_table['90'].iloc[v]
        elif column == 0.95:
            quantile = self.t_table['95'].iloc[v]
        elif column == 0.975:
            quantile = self.t_table['975'].iloc[v]
        elif column == 0.995:
            quantile = self.t_table['995'].iloc[v]
        elif column == 0.9995:
            quantile = self.t_table['9995'].iloc[v]

        return quantile