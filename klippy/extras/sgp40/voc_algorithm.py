# SPDX-FileCopyrightText: Copyright (c) 2010 DFRobot Co.Ltd
# (http://www.dfrobot.com)
#
# SPDX-License-Identifier: MIT
"""
`voc_algorithm`
================================================================================

Algorithm to convert Sensirion sgp40 raw reading to indexed voc readings.


* Author(s): yangfeng

"""

_SAMPLING_INTERVAL = 1
_INITIAL_BLACKOUT = 45
_VOC_INDEX_GAIN = 230
_SRAW_STD_INITIAL = 50
_SRAW_STD_BONUS = 220
_TAU_MEAN_VARIANCE_HOURS = 12
_TAU_INITIAL_MEAN = 20
_INITI_DURATION_MEAN = 2700
_INITI_TRANSITION_MEAN = 0.01
_TAU_INITIAL_VARIANCE = 2500
_INITI_DURATION_VARIANCE = 5220
_INITI_TRANSITION_VARIANCE = 0.01
_GATING_THRESHOLD = 340
_GATING_THRESHOLD_INITIAL = 510
_GATING_THRESHOLD_TRANSITION = 0.09
_GATING_MAX_DURATION_MINUTES = 180
_GATING_MAX_RATIO = 0.3
_SIGMOID_L = 500
_SIGMOID_K = -0.0065
_SIGMOID_X0 = 213
_VOC_INDEX_OFFSET_DEFAULT = 100
_LP_TAU_FAST = 20
_LP_TAU_SLOW = 500
_LP_ALPHA = -0.2
_PERSISTENCE_UPTIME_GAMMA = 10800
_MVE_GAMMA_SCALING = 64
_MVE_FIX16_MAX = 32767
_FIX16_MAXIMUM = 0x7FFFFFFF
_FIX16_MINIMUM = 0x80000000
_FIX16_OVERFLOW = 0x80000000
_FIX16_ONE = 0x00010000


class VOCAlgorithm:
    def __init__(self):
        self.mvoc_index_offset = 0
        self.mtau_mean_variance_hours = 0
        self.mgating_max_duration_minutes = 0
        self.msraw_std_initial = 0
        self.muptime = 0
        self.msraw = 0
        self.mvoc_index = 0
        self.mve_gating_max_duration_minutes = 0
        self.mve_initialized = 0
        self.mve_mean = 0
        self.mve_sraw_offset = 0
        self.mve_std = 0
        self.mve_gamma = 0
        self.mve_gamma_initial_mean = 0
        self.mve_gamma_initial_variance = 0
        self.mve_gamma_mean = 0
        self.mve__gamma_variance = 0
        self.mve_uptime_gamma = 0
        self.mve_uptime_gating = 0
        self.mve_gating_duration_minutes = 0
        self.mve_sigmoid_l = 0
        self.mve_sigmoid_k = 0
        self.mve_sigmoid_x0 = 0
        self.mox_model_sraw_mean = 0
        self.sigmoid_scaled_offset = 0
        self.adaptive_lowpass_a1 = 0
        self.adaptive_lowpass_a2 = 0
        self.adaptive_lowpass_initialized = 0
        self.adaptive_lowpass_x1 = 0
        self.adaptive_lowpass_x2 = 0
        self.adaptive_lowpass_x3 = 0
        self.vocalgorithm_init()

    def _f16(self, x):
        if x >= 0:
            return int((x) * 65536.0 + 0.5)
        else:
            return int((x) * 65536.0 - 0.5)

    def _fix16_from_int(self, a):
        return int(a * _FIX16_ONE)

    def _fix16_cast_to_int(self, a):
        return int(a) >> 16

    def _fix16_mul(self, inarg0, inarg1):
        inarg0 = int(inarg0)
        inarg1 = int(inarg1)
        A = inarg0 >> 16
        if inarg0 < 0:
            B = (inarg0 & 0xFFFFFFFF) & 0xFFFF
        else:
            B = inarg0 & 0xFFFF
        C = inarg1 >> 16
        if inarg1 < 0:
            D = (inarg1 & 0xFFFFFFFF) & 0xFFFF
        else:
            D = inarg1 & 0xFFFF
        AC = A * C
        AD_CB = A * D + C * B
        BD = B * D
        product_hi = AC + (AD_CB >> 16)
        ad_cb_temp = ((AD_CB) << 16) & 0xFFFFFFFF
        product_lo = ((BD + ad_cb_temp)) & 0xFFFFFFFF
        if product_lo < BD:
            product_hi = product_hi + 1
        if (product_hi >> 31) != (product_hi >> 15):
            return _FIX16_OVERFLOW
        product_lo_tmp = product_lo & 0xFFFFFFFF
        product_lo = (product_lo - 0x8000) & 0xFFFFFFFF
        product_lo = (
            (product_lo - ((product_hi & 0xFFFFFFFF) >> 31))
            & 0xFFFFFFFF
        )
        if product_lo > product_lo_tmp:
            product_hi = product_hi - 1
        result = (product_hi << 16) | (product_lo >> 16)
        result += 1
        return result

    def _fix16_div(self, a, b):
        a = int(a)
        b = int(b)
        if b == 0:
            return _FIX16_MINIMUM
        if a >= 0:
            remainder = a
        else:
            remainder = (a * (-1)) & 0xFFFFFFFF
        if b >= 0:
            divider = b
        else:
            divider = (b * (-1)) & 0xFFFFFFFF
        quotient = 0
        bit = 0x10000
        while divider < remainder:
            divider = divider << 1
            bit <<= 1
        if not bit:
            return _FIX16_OVERFLOW
        if divider & 0x80000000:
            if remainder >= divider:
                quotient |= bit
                remainder -= divider
            divider >>= 1
            bit >>= 1
        while bit and remainder:
            if remainder >= divider:
                quotient |= bit
                remainder -= divider
            remainder <<= 1
            bit >>= 1
        if remainder >= divider:
            quotient += 1
        result = quotient
        if (a ^ b) & 0x80000000:
            if result == _FIX16_MINIMUM:
                return _FIX16_OVERFLOW
            result = -result
        return result

    def _fix16_sqrt(self, x):
        x = int(x)
        num = x & 0xFFFFFFFF
        result = 0
        bit = 1 << 30
        while bit > num:
            bit >>= 2
        for n in range(0, 2):
            while bit:
                if num >= result + bit:
                    num = num - (result + bit) & 0xFFFFFFFF
                    result = (result >> 1) + bit
                else:
                    result = result >> 1
                bit >>= 2
            if n == 0:
                if num > 65535:
                    num = (num - result) & 0xFFFFFFFF
                    num = ((num << 16) - 0x8000) & 0xFFFFFFFF
                    result = ((result << 16) + 0x8000) & 0xFFFFFFFF
                else:
                    num = (num << 16) & 0xFFFFFFFF
                    result = (result << 16) & 0xFFFFFFFF
                bit = 1 << 14
        if num > result:
            result += 1
        return result

    def _fix16_exp(self, x):
        x = int(x)
        exp_pos_values = [
            self._f16(2.7182818),
            self._f16(1.1331485),
            self._f16(1.0157477),
            self._f16(1.0019550),
        ]
        exp_neg_values = [
            self._f16(0.3678794),
            self._f16(0.8824969),
            self._f16(0.9844964),
            self._f16(0.9980488),
        ]
        if x >= self._f16(10.3972):
            return _FIX16_MAXIMUM
        if x <= self._f16(-11.7835):
            return 0
        if x < 0:
            x = -x
            exp_values = exp_neg_values
        else:
            exp_values = exp_pos_values
        res = _FIX16_ONE
        arg = _FIX16_ONE
        for i in range(0, 4):
            while x >= arg:
                res = self._fix16_mul(res, exp_values[i])
                x -= arg
            arg >>= 3
        return res

    def vocalgorithm_init(self):
        self.mvoc_index_offset = self._f16(
            _VOC_INDEX_OFFSET_DEFAULT
        )
        self.mtau_mean_variance_hours = self._f16(
            _TAU_MEAN_VARIANCE_HOURS
        )
        self.mgating_max_duration_minutes = self._f16(
            _GATING_MAX_DURATION_MINUTES
        )
        self.msraw_std_initial = self._f16(_SRAW_STD_INITIAL)
        self.muptime = self._f16(0.0)
        self.msraw = self._f16(0.0)
        self.mvoc_index = 0
        self._init_instances()

    def _init_instances(self):
        self._mean_variance_estimator__init()
        self._mean_variance_estimator__set_parameters(
            self._f16(_SRAW_STD_INITIAL),
            self.mtau_mean_variance_hours,
            self.mgating_max_duration_minutes,
        )
        self._mox_model__init()
        self._mox_model__set_parameters(
            self._mean_variance_estimator__get_std(),
            self._mean_variance_estimator__get_mean(),
        )
        self._sigmoid_scaled__init()
        self._sigmoid_scaled__set_parameters(
            self.mvoc_index_offset
        )
        self._adaptive_lowpass__init()
        self._adaptive_lowpass__set_parameters()

    def _vocalgorithm_get_states(self, state0, state1):
        state0 = self._mean_variance_estimator__get_mean()
        state1 = self._mean_variance_estimator__get_std()
        return state0, state1

    def _vocalgorithm_set_states(self, state0, state1):
        self._mean_variance_estimator__set_states(
            state0,
            state1,
            self._f16(_PERSISTENCE_UPTIME_GAMMA),
        )
        self.msraw = state0

    def _vocalgorithm_set_tuning_parameters(
        self,
        voc_index_offset,
        learning_time_hours,
        gating_max_duration_minutes,
        std_initial,
    ):
        self.mvoc_index_offset = self._fix16_from_int(voc_index_offset)
        self.mtau_mean_variance_hours = self._fix16_from_int(
            learning_time_hours
        )
        self.mgating_max_duration_minutes = self._fix16_from_int(
            gating_max_duration_minutes
        )
        self.msraw_std_initial = self._fix16_from_int(std_initial)
        self._init_instances()

    def process(self, sraw):
        if self.muptime <= self._f16(_INITIAL_BLACKOUT):
            self.muptime = self.muptime + self._f16(
                _SAMPLING_INTERVAL
            )
        else:
            if (sraw > 0) and (sraw < 65000):
                if sraw < 20001:
                    sraw = 20001
                elif sraw > 52767:
                    sraw = 52767
                self.msraw = self._fix16_from_int((sraw - 20000))
            self.mvoc_index = self._mox_model__process(
                self.msraw
            )
            self.mvoc_index = self._sigmoid_scaled__process(
                self.mvoc_index
            )
            self.mvoc_index = self._adaptive_lowpass__process(
                self.mvoc_index
            )
            if self.mvoc_index < self._f16(0.5):
                self.mvoc_index = self._f16(0.5)
            if self.msraw > self._f16(0.0):
                self._mean_variance_estimator__process(
                    self.msraw, self.mvoc_index
                )
                self._mox_model__set_parameters(
                    self._mean_variance_estimator__get_std(),
                    self._mean_variance_estimator__get_mean(),
                )
        voc_index = self._fix16_cast_to_int((self.mvoc_index + self._f16(0.5)))
        return voc_index

    def _mean_variance_estimator__init(self):
        self._mean_variance_estimator__set_parameters(
            self._f16(0.0), self._f16(0.0), self._f16(0.0)
        )
        self._mean_variance_estimator___init_instances()

    def _mean_variance_estimator___init_instances(self):
        self._mean_variance_estimator___sigmoid__init()

    def _mean_variance_estimator__set_parameters(
        self,
        std_initial,
        tau_mean_variance_hours,
        gating_max_duration_minutes,
    ):
        self.mve_gating_max_duration_minutes = (
            gating_max_duration_minutes
        )
        self.mve_initialized = 0
        self.mve_mean = self._f16(0.0)
        self.mve_sraw_offset = self._f16(0.0)
        self.mve_std = std_initial
        self.mve_gamma = self._fix16_div(
            self._f16(
                (
                    _MVE_GAMMA_SCALING
                    * (_SAMPLING_INTERVAL / 3600.0)
                )
            ),
            (
                tau_mean_variance_hours
                + self._f16((_SAMPLING_INTERVAL / 3600.0))
            ),
        )
        self.mve_gamma_initial_mean = self._f16(
            (
                (
                    _MVE_GAMMA_SCALING
                    * _SAMPLING_INTERVAL
                )
                / (_TAU_INITIAL_MEAN + _SAMPLING_INTERVAL)
            )
        )
        self.mve_gamma_initial_variance = self._f16(
            (
                (
                    _MVE_GAMMA_SCALING
                    * _SAMPLING_INTERVAL
                )
                / (_TAU_INITIAL_VARIANCE + _SAMPLING_INTERVAL)
            )
        )
        self.mve_gamma_mean = self._f16(0.0)
        self.mve__gamma_variance = self._f16(0.0)
        self.mve_uptime_gamma = self._f16(0.0)
        self.mve_uptime_gating = self._f16(0.0)
        self.mve_gating_duration_minutes = self._f16(0.0)

    def _mean_variance_estimator__set_states(self, mean, std, uptime_gamma):
        self.mve_mean = mean
        self.mve_std = std
        self.mve_uptime_gamma = uptime_gamma
        self.mve_initialized = True

    def _mean_variance_estimator__get_std(self):
        return self.mve_std

    def _mean_variance_estimator__get_mean(self):
        return (
            self.mve_mean
            + self.mve_sraw_offset
        )

    def _mean_variance_estimator___calculate_gamma(self, voc_index_from_prior):
        uptime_limit = self._f16(
            (
                _MVE_FIX16_MAX
                - _SAMPLING_INTERVAL
            )
        )
        if self.mve_uptime_gamma < uptime_limit:
            self.mve_uptime_gamma = (
                self.mve_uptime_gamma
                + self._f16(_SAMPLING_INTERVAL)
            )

        if self.mve_uptime_gating < uptime_limit:
            self.mve_uptime_gating = (
                self.mve_uptime_gating
                + self._f16(_SAMPLING_INTERVAL)
            )

        self._mean_variance_estimator___sigmoid__set_parameters(
            self._f16(1.0),
            self._f16(_INITI_DURATION_MEAN),
            self._f16(_INITI_TRANSITION_MEAN),
        )
        sigmoid_gamma_mean = (
            self._mean_variance_estimator___sigmoid__process(
                self.mve_uptime_gamma
            )
        )
        gamma_mean = self.mve_gamma + (
            self._fix16_mul(
                (
                    self.mve_gamma_initial_mean
                    - self.mve_gamma
                ),
                sigmoid_gamma_mean,
            )
        )
        gating_threshold_mean = self._f16(_GATING_THRESHOLD) + (
            self._fix16_mul(
                self._f16(
                    (
                        _GATING_THRESHOLD_INITIAL
                        - _GATING_THRESHOLD
                    )
                ),
                self._mean_variance_estimator___sigmoid__process(
                    self.mve_uptime_gating
                ),
            )
        )
        self._mean_variance_estimator___sigmoid__set_parameters(
            self._f16(1.0),
            gating_threshold_mean,
            self._f16(_GATING_THRESHOLD_TRANSITION),
        )

        sigmoid_gating_mean = (
            self._mean_variance_estimator___sigmoid__process(
                voc_index_from_prior
            )
        )
        self.mve_gamma_mean = self._fix16_mul(
            sigmoid_gating_mean, gamma_mean
        )

        self._mean_variance_estimator___sigmoid__set_parameters(
            self._f16(1.0),
            self._f16(_INITI_DURATION_VARIANCE),
            self._f16(_INITI_TRANSITION_VARIANCE),
        )

        sigmoid_gamma_variance = (
            self._mean_variance_estimator___sigmoid__process(
                self.mve_uptime_gamma
            )
        )

        gamma_variance = self.mve_gamma + (
            self._fix16_mul(
                (
                    self.mve_gamma_initial_variance
                    - self.mve_gamma
                ),
                (sigmoid_gamma_variance - sigmoid_gamma_mean),
            )
        )

        gating_threshold_variance = self._f16(_GATING_THRESHOLD) + (
            self._fix16_mul(
                self._f16(
                    (
                        _GATING_THRESHOLD_INITIAL
                        - _GATING_THRESHOLD
                    )
                ),
                self._mean_variance_estimator___sigmoid__process(
                    self.mve_uptime_gating
                ),
            )
        )

        self._mean_variance_estimator___sigmoid__set_parameters(
            self._f16(1.0),
            gating_threshold_variance,
            self._f16(_GATING_THRESHOLD_TRANSITION),
        )

        sigmoid_gating_variance = (
            self._mean_variance_estimator___sigmoid__process(
                voc_index_from_prior
            )
        )

        self.mve__gamma_variance = self._fix16_mul(
            sigmoid_gating_variance, gamma_variance
        )

        self.mve_gating_duration_minutes = (
            self.mve_gating_duration_minutes
            + (
                self._fix16_mul(
                    self._f16((_SAMPLING_INTERVAL / 60.0)),
                    (
                        (
                            self._fix16_mul(
                                (self._f16(1.0) - sigmoid_gating_mean),
                                self._f16((1.0 + _GATING_MAX_RATIO)),
                            )
                        )
                        - self._f16(_GATING_MAX_RATIO)
                    ),
                )
            )
        )

        if self.mve_gating_duration_minutes < self._f16(
            0.0
        ):
            self.mve_gating_duration_minutes = self._f16(
                0.0
            )

        if (
            self.mve_gating_duration_minutes
            > self.mve_gating_max_duration_minutes
        ):
            self.mve_uptime_gating = self._f16(0.0)

    def _mean_variance_estimator__process(self, sraw, voc_index_from_prior):
        if self.mve_initialized == 0:
            self.mve_initialized = 1
            self.mve_sraw_offset = sraw
            self.mve_mean = self._f16(0.0)
        else:
            if (self.mve_mean >= self._f16(100.0)) or (
                self.mve_mean <= self._f16(-100.0)
            ):
                self.mve_sraw_offset = (
                    self.mve_sraw_offset
                    + self.mve_mean
                )
                self.mve_mean = self._f16(0.0)

            sraw = sraw - self.mve_sraw_offset
            self._mean_variance_estimator___calculate_gamma(
                voc_index_from_prior
            )
            delta_sgp = self._fix16_div(
                (sraw - self.mve_mean),
                self._f16(_MVE_GAMMA_SCALING),
            )
            if delta_sgp < self._f16(0.0):
                c = self.mve_std - delta_sgp
            else:
                c = self.mve_std + delta_sgp
            additional_scaling = self._f16(1.0)
            if c > self._f16(1440.0):
                additional_scaling = self._f16(4.0)
            self.mve_std = self._fix16_mul(
                self._fix16_sqrt(
                    (
                        self._fix16_mul(
                            additional_scaling,
                            (
                                self._f16(
                                    _MVE_GAMMA_SCALING
                                )
                                - self.mve__gamma_variance
                            ),
                        )
                    )
                ),
                self._fix16_sqrt(
                    (
                        (
                            self._fix16_mul(
                                self.mve_std,
                                (
                                    self._fix16_div(
                                        self.mve_std,
                                        (
                                            self._fix16_mul(
                                                self._f16(
                                                    _MVE_GAMMA_SCALING
                                                ),
                                                additional_scaling,
                                            )
                                        ),
                                    )
                                ),
                            )
                        )
                        + (
                            self._fix16_mul(
                                (
                                    self._fix16_div(
                                        (
                                            self._fix16_mul(
                                                self.mve__gamma_variance,
                                                delta_sgp,
                                            )
                                        ),
                                        additional_scaling,
                                    )
                                ),
                                delta_sgp,
                            )
                        )
                    )
                ),
            )
            self.mve_mean = (
                self.mve_mean
                + (
                    self._fix16_mul(
                        self.mve_gamma_mean, delta_sgp
                    )
                )
            )

    def _mean_variance_estimator___sigmoid__init(self):
        self._mean_variance_estimator___sigmoid__set_parameters(
            self._f16(0.0), self._f16(0.0), self._f16(0.0)
        )

    def _mean_variance_estimator___sigmoid__set_parameters(self, l, x0, k):
        self.mve_sigmoid_l = l
        self.mve_sigmoid_k = k
        self.mve_sigmoid_x0 = x0

    def _mean_variance_estimator___sigmoid__process(self, sample):
        x = self._fix16_mul(
            self.mve_sigmoid_k,
            (sample - self.mve_sigmoid_x0),
        )
        if x < self._f16(-50.0):
            return self.mve_sigmoid_l
        elif x > self._f16(50.0):
            return self._f16(0.0)
        else:
            return self._fix16_div(
                self.mve_sigmoid_l,
                (self._f16(1.0) + self._fix16_exp(x)),
            )

    def _mox_model__init(self):
        self._mox_model__set_parameters(self._f16(1.0), self._f16(0.0))

    def _mox_model__set_parameters(self, sraw_std, sraw_mean):
        self.mox_model_sraw_std = sraw_std
        self.mox_model_sraw_mean = sraw_mean

    def _mox_model__process(self, sraw):
        return self._fix16_mul(
            (
                self._fix16_div(
                    (sraw - self.mox_model_sraw_mean),
                    (
                        -(
                            self.mox_model_sraw_std
                            + self._f16(_SRAW_STD_BONUS)
                        )
                    ),
                )
            ),
            self._f16(_VOC_INDEX_GAIN),
        )

    def _sigmoid_scaled__init(self):
        self._sigmoid_scaled__set_parameters(self._f16(0.0))

    def _sigmoid_scaled__set_parameters(self, offset):
        self.sigmoid_scaled_offset = offset

    def _sigmoid_scaled__process(self, sample):
        x = self._fix16_mul(
            self._f16(_SIGMOID_K),
            (sample - self._f16(_SIGMOID_X0)),
        )
        if x < self._f16(-50.0):
            return self._f16(_SIGMOID_L)
        elif x > self._f16(50.0):
            return self._f16(0.0)
        else:
            if sample >= self._f16(0.0):
                shift = self._fix16_div(
                    (
                        self._f16(_SIGMOID_L)
                        - (
                            self._fix16_mul(
                                self._f16(5.0), self.sigmoid_scaled_offset
                            )
                        )
                    ),
                    self._f16(4.0),
                )
                return (
                    self._fix16_div(
                        (self._f16(_SIGMOID_L) + shift),
                        (self._f16(1.0) + self._fix16_exp(x)),
                    )
                ) - shift
            else:
                return self._fix16_mul(
                    (
                        self._fix16_div(
                            self.sigmoid_scaled_offset,
                            self._f16(_VOC_INDEX_OFFSET_DEFAULT),
                        )
                    ),
                    (
                        self._fix16_div(
                            self._f16(_SIGMOID_L),
                            (self._f16(1.0) + self._fix16_exp(x)),
                        )
                    ),
                )

    def _adaptive_lowpass__init(self):
        self._adaptive_lowpass__set_parameters()

    def _adaptive_lowpass__set_parameters(self):
        self.adaptive_lowpass_a1 = self._f16(
            (
                _SAMPLING_INTERVAL
                / (_LP_TAU_FAST + _SAMPLING_INTERVAL)
            )
        )
        self.adaptive_lowpass_a2 = self._f16(
            (
                _SAMPLING_INTERVAL
                / (_LP_TAU_SLOW + _SAMPLING_INTERVAL)
            )
        )
        self.adaptive_lowpass_initialized = 0

    def _adaptive_lowpass__process(self, sample):
        if self.adaptive_lowpass_initialized == 0:
            self.adaptive_lowpass_x1 = sample
            self.adaptive_lowpass_x2 = sample
            self.adaptive_lowpass_x3 = sample
            self.adaptive_lowpass_initialized = 1
        self.adaptive_lowpass_x1 = (
            self._fix16_mul(
                (self._f16(1.0) - self.adaptive_lowpass_a1),
                self.adaptive_lowpass_x1,
            )
        ) + (self._fix16_mul(self.adaptive_lowpass_a1, sample))

        self.adaptive_lowpass_x2 = (
            self._fix16_mul(
                (self._f16(1.0) - self.adaptive_lowpass_a2),
                self.adaptive_lowpass_x2,
            )
        ) + (self._fix16_mul(self.adaptive_lowpass_a2, sample))

        abs_delta = (
            self.adaptive_lowpass_x1 - self.adaptive_lowpass_x2
        )

        if abs_delta < self._f16(0.0):
            abs_delta = -abs_delta
        F1 = self._fix16_exp(
            (self._fix16_mul(self._f16(_LP_ALPHA), abs_delta))
        )
        tau_a = (
            self._fix16_mul(
                self._f16((_LP_TAU_SLOW - _LP_TAU_FAST)), F1
            )
        ) + self._f16(_LP_TAU_FAST)
        a3 = self._fix16_div(
            self._f16(_SAMPLING_INTERVAL),
            (self._f16(_SAMPLING_INTERVAL) + tau_a),
        )
        self.adaptive_lowpass_x3 = (
            self._fix16_mul((self._f16(1.0) - a3), self.adaptive_lowpass_x3)
        ) + (self._fix16_mul(a3, sample))
        return self.adaptive_lowpass_x3
