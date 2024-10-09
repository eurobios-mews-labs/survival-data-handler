# Copyright 2024 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from survival_data_handler.utils import compute_derivative, shift_from_interp, residual_life


class SurvivalCurves:

    def __init__(self, curve):
        self.__curve = TimeCurveData(curve)

    @property
    def survival_function(self):
        self.__curve[self.__curve > 1] = 1
        self.__curve[self.__curve < 0] = 0
        return self.__curve

    @property
    def density_function(self):
        return - self.survival_function.derivative()

    @property
    def hazard_function(self):
        return self.density_function / self.survival_function

    @property
    def cumulative_hazard_function(self):
        return - self.survival_function.log()

    @property
    def residual_life(self):
        ret = residual_life(self.survival_function)
        ret.columns = self.survival_function.columns
        return TimeCurveData(ret)

    @property
    def lifetime_distribution_function(self):
        return 1 - self.survival_function


class TimeCurve:

    def __interpolator(self):
        pass


class TimeCurveData(pd.DataFrame, TimeCurve):
    def __init__(self, *args):
        super().__init__(*args)
        self.__checks()

    def __checks(self):
        if not all(self.columns[i] <= self.columns[i + 1] for i in range(len(self.columns) - 1)):
            raise ValueError(
                "Columns must be sorted")
        # if not isinstance(self.columns[0], pd.Timedelta):
        #     raise ValueError("Columns must symbolize time index")

    def __interpolator(self):
        self.__interpolation = {
            c: interp1d(self.columns.astype("int64"), self.loc[c], fill_value="extrapolate")
            for c in self.index}

    def derivative(self):
        return TimeCurveData(compute_derivative(self))

    def log(self):
        return TimeCurveData(np.log(self))

    def exp(self):
        return TimeCurveData(np.exp(self))

    def __truediv__(self, other):
        return TimeCurveData(
            pd.DataFrame(
                self.values / self.__other(other),
                columns=self.columns,
                index=self.index))

    def __add__(self, other):
        return TimeCurveData(
            pd.DataFrame(
                self.values + self.__other(other),
                columns=self.columns,
                index=self.index))

    def __sub__(self, other):
        return TimeCurveData(
            pd.DataFrame(
                self.values - self.__other(other),
                columns=self.columns,
                index=self.index))

    def __neg__(self):
        return TimeCurveData(pd.DataFrame(-self.values, index=self.index, columns=self.columns))

    @property
    def interpolation(self) -> dict:
        if not hasattr(self, "__interpolation"):
            self.__interpolator()
        return self.__interpolation

    @staticmethod
    def __other(other):
        if hasattr(other, "values"):
            return other.values
        else:
            return other


class TimeCurveInterpolation(TimeCurve):

    def __init__(
            self,
            interpolation: dict,
            birth: iter,
            index: pd.Series,
            window: tuple,
            period):
        self.__interpolation = interpolation

        self.__matching = pd.DataFrame()
        self.__matching["index"] = index
        self.__matching["birth"] = birth
        self.__matching.index = index.index
        self.__matching.index.name = 'origin_index'

        self.__reduced_matching = self.__matching.drop_duplicates(
            subset=["birth", "index"]
        )
        self.__period = period
        self.__window = window

    @property
    def interpolation(self) -> pd.Series:
        return pd.Series(self.__interpolation, name="interpolation")

    def __shift(self) -> TimeCurveData:
        data = pd.merge(
            self.__reduced_matching["index"].reset_index(),
            self.interpolation,
            right_index=True,
            left_on='index')

        data = data.set_index("origin_index").drop("index", axis=1)
        ret = shift_from_interp(
            data=data,
            starting_dates=self.__reduced_matching["birth"],
            period=self.__period,
            date_initial=self.__window[0],
            date_final=self.__window[1],
            dtypes="float")
        self.__curve = TimeCurveData(ret)
        return self.__curve

    @property
    def unique_curve(self) -> TimeCurveData:
        if not hasattr(self, "__curve"):
            self.__shift()
        return self.__curve

    @property
    def curve(self) -> TimeCurveData:
        ret = pd.merge(
            self.__reduced_matching, self.unique_curve,
            right_index=True,
            left_index=True)
        m = self.__matching
        ret = pd.merge(m, ret, on=["birth", "index"], how="left")
        ret.index = m.index
        ret = ret[self.unique_curve.columns]
        return TimeCurveData(ret)
