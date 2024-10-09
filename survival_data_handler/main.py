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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from survival_data_handler import plotting
from survival_data_handler.base import TimeCurveData, SurvivalCurves, TimeCurveInterpolation
from survival_data_handler.utils import process_survival_function

# ======================================================================================================================
# AESTHETICS
cmap = sns.color_palette('rainbow', as_cmap=True)
event_censored_color = cmap(0.1)
event_observed_color = cmap(0.95)


class Lifespan(SurvivalCurves):
    __index = "origin_index"

    def __init__(
            self,
            ageing_curves: pd.DataFrame,
            index: iter,
            window: tuple,
            age: iter = None,
            birth: iter = None,
            default_precision="float",
            time_delta=pd.to_timedelta(30, "D")):
        self.__age = age
        self.__birth = birth
        self.__curves = TimeCurveData(ageing_curves.__deepcopy__())

        self.estimations = SurvivalEstimation(self.curves)
        self.__tci = TimeCurveInterpolation(
            self.estimations.survival_function.interpolation,
            birth,
            index,
            window,
            time_delta
        )

        self.confusion_matrix_threshold = np.nan
        self.__supervised = False
        self.__interpolator = {}
        self.__check_args()
        super().__init__(self.__tci.curve)

    def __check_args(self):
        if hasattr(self.curves.columns, "dt"):
            raise ValueError("curves index must have dt property")
        if self.__age is None and self.__birth is None:
            raise ValueError("age or birth : one of them must be set")

    @property
    def supervised(self) -> bool:
        return self.__supervised

    def add_supervision(self, event, durations, entry=None):
        self.event = event
        self.durations = durations
        self.entry = entry
        self.__supervised = True

    def __get_align_data(self, on, get_supervision=False):
        ret = self.__getattribute__(on)
        if not get_supervision:
            return ret, None, None, None
        else:
            index = [i for i in ret.index if i in self.event.index]
            entry = self.entry if self.entry is None else self.entry.loc[index]
            return ret.loc[index], self.event.loc[index], self.durations.loc[index], entry

    # ==============================================
    #           ENRICH REPRESENTATION
    # ==============================================

    def residual_survival(self, t0) -> pd.DataFrame:

        t0 = max(t0, self.survival_function.columns[0])
        s0 = _get(self.survival_function, t0).astype("float16").values
        df = self.survival_function.divide(s0, axis=0).copy()
        df = df[[c for c in df.columns if c > t0]]
        return df

    def compute_times(self, p=0.95, on="survival_function"):
        cond_low = (self.__getattribute__(on) > p).sum(axis=1)
        dates = self.__getattribute__(on).columns.to_numpy()
        return pd.Series(dates[cond_low])

    def remove_the_dead(self, on, error="coerce"):
        data = self.__getattribute__(on)
        try:
            data = data.loc[self.event.index[~self.event.astype(bool)]]
        except KeyError as e:
            print(e)

            if error == "raise":
                raise KeyError(e)
            elif error == "coerce":
                index = [i for i in self.event.index[~self.event.astype(bool)] if i in data.index]
                data = data.loc[index]

        return data

    def percentile_life(self, q: float = 0.5) -> pd.DataFrame:
        """
        Compute the percentile lifetime corresponding to the proportion of
        the population (in terms of curves) that remains
        """
        min_ = pd.DataFrame(self.survival_function > q)
        cs = min_.cumsum(axis=1)
        df = pd.DataFrame(np.array(cs.columns.to_list())[cs.iloc[:, -1].values - 1], columns=["percentile_life_%s" % q])
        del min_, cs
        return df

    def compute_confusion_matrix(self, on: str, threshold: float, score=True):
        from sklearn import metrics
        data = self.__getattribute__(on)
        if score:
            data = 1 - data

        time_start, time_stop = self.durations.min(), self.durations.max()
        self.confusion_matrix_threshold = threshold
        predicted_events = pd.DataFrame(data > threshold)[[c for c in data.columns if time_start <= c <= time_stop]]

        real_life_events = predicted_events.__deepcopy__()
        nan = data.isna()[[c for c in data.columns if time_start <= c <= time_stop]]

        df_matrix = pd.DataFrame(index=real_life_events.columns, columns=["TN", "FP", "FN", "TP"])
        for c in real_life_events.columns:
            real_life_events[c] = (self.event.astype(bool) & (self.durations < c))
            args = real_life_events[c][~nan[c]].astype(bool), predicted_events[c][~nan[c]].astype(bool)

            df_matrix.loc[c] = metrics.confusion_matrix(*args).ravel()
        df_matrix = df_matrix.astype(float)
        tn, fp, fn, tp = df_matrix["TN"], df_matrix["FP"], df_matrix["FN"], df_matrix["TP"]

        df_matrix["P"] = p = tp + fn
        df_matrix["N"] = n = tn + fp
        df_matrix["Total"] = p + n
        df_matrix["accuracy"] = (tp + tn) / (p + n)

        df_matrix["precision"] = precision = np.divide(tp, tp + fp)

        df_matrix["recall"] = recall = np.divide(tp, tp + fn)

        df_matrix["f1-score"] = 2 * recall * precision / (recall + precision)
        df_matrix["f2-score"] = (1 + 2) ** 2 * recall * precision / (2 ** 2 * precision + recall)

        return df_matrix

    # ==============================================
    #           PLOTS CURVES
    # ==============================================
    def plot_curves_residual_life(self, **kwargs):
        self.estimations.plot_residual_life(**kwargs)
        plt.ylabel("Expected residual lifespan")
        plt.xlabel("Time [Y]")

    def plot_average_tagged(self, on: str, event_type=None, plot_test_window=False, plot_type="dist"):
        corresp, event, duration, entry = self.__get_align_data(on, get_supervision=True)
        if event_type == "censored":
            sample = corresp[~event.astype(bool)].astype("float32")
            color = event_censored_color
        elif event_type == "observed":
            sample = corresp[event.astype(bool)].astype("float32")
            color = event_observed_color

        else:
            sample = corresp.astype("float32")
            color = event_censored_color

        if plot_type == "dist":
            percentiles = np.linspace(0, 1, 21)
            sample_des = sample.describe(percentiles=percentiles)

            def ts(__p):
                return str(int(np.round(__p, 3) * 100)) + "%"

            for p in percentiles:
                label = None if p != percentiles[0] else f"{event_type} event"
                plt.fill_between(sample_des.columns, sample_des.loc[ts(p)], sample_des.loc[ts(1 - p)], color=color,
                                 alpha=0.1, label=label)
        else:
            sample_des = sample.describe()

            plt.plot(sample_des.columns, sample_des.loc["50%"], color=event_censored_color, label=f"{event_type} event")

            plt.fill_between(sample_des.columns, sample_des.loc["25%"], sample_des.loc["75%"], color=color, alpha=0.2)

        if plot_test_window:
            t0 = duration[event.astype(bool)].min()
            t1 = duration[event.astype(bool)].max()

            plt.axvline(t0, lw=1.5, ls="--", color="k")
            plt.axvline(t1, lw=1.5, ls='--', color="k", label="Test window")
        plt.legend()

    def plot_dist_facet_grid(self, on: str, n=4):
        bins = np.linspace(0, 1, 50)
        data, event, duration, entry = self.__get_align_data(on, get_supervision=True)
        s = data.shape[1]
        dates = data.columns.to_numpy()[np.array([(s * i) // n for i in range(1, n)])]

        df = data[dates].unstack().reset_index()
        data["event"] = event
        df = pd.merge(df, data.reset_index()[["origin_index", "event"]], on="origin_index")
        df = df.rename({"level_0": "dates", 0: "survival"}, axis=1)
        df["dates"] = df["dates"].dt.year

        g = sns.FacetGrid(df, row="dates", hue="event", aspect=5, height=2, palette="RdBu_r")
        g.map(sns.histplot, "survival", fill=False, alpha=1, linewidth=2, bins=bins, element="step",  # cumulative=True,
              stat="density", common_norm=False)
        plt.legend()

    def plot_tagged_sample(self, on: str, n_sample=None, n_sample_pos=None, n_sample_neg=None, ):

        assert self.__supervised, "No supervision provided"
        data, event, duration, entry = self.__get_align_data(on, get_supervision=True)
        if n_sample is None and n_sample_neg is None and n_sample_pos is None:
            sample = data
        elif n_sample_pos is None and n_sample_neg is None:
            sample = data.sample(n_sample)
        elif n_sample is None:
            pos_sample = data[event.astype(bool)].sample(n_sample_pos)
            neg_sample = data[~event.astype(bool)].sample(n_sample_neg)
            sample = pd.concat((neg_sample, pos_sample), axis=0)
        else:
            raise ValueError("Error in argument specifications")
        index = sample.index

        plotting.tagged_curves(temporal_curves=sample, label=event.loc[index], time_event=duration.loc[index],
                               event_censored_color=event_censored_color, event_observed_color=event_observed_color,
                               add_marker=True, )
        return index

    def plotly_auc_vs_score(self, on: str, temporal_metric: pd.Series, **kwargs):
        corresp, event, duration, entry = self.__get_align_data(on, get_supervision=True)
        time_start, time_stop = duration.min(), duration.max()
        corresp = corresp[[c for c in corresp.columns if time_start <= c <= time_stop]]
        return plotting.auc_vs_score_plotly(temporal_scores=corresp, event_observed=event, censoring_time=duration,
                                            entry_time=entry, temporal_metric=temporal_metric, **kwargs)

    def plotly_roc_curve(self, on, method="roc-cd", colormap="magma_r", template="plotly_white", **kwargs):
        roc = self.assess_metric(on, method, metric=metrics.roc_curve)
        return plotting.plot_roc_curve_plotly(roc, colormap=colormap, template=template, **kwargs)

    # ==============================================
    #           METRIC ASSESSMENT
    # ==============================================
    def assess_metric(self, on, method="roc-cd", metric=metrics.roc_auc_score):
        from survival_data_handler.metric import time_dependent_helper
        data, event, duration, entry = self.__get_align_data(on, get_supervision=True)

        time_start, time_stop = duration.min(), duration.max()
        data = data[[c for c in data.columns if time_start <= c <= time_stop]]
        return pd.Series(time_dependent_helper(
            data,
            event_observed=event,
            censoring_time=duration,
            method=method,
            function=metric))

    # ==============================================
    #           Properties
    # ==============================================
    @property
    def curves(self) -> TimeCurveData:
        return self.__curves


def _get(data: pd.DataFrame, date):
    search = np.searchsorted(data.columns.to_list(), date)
    return data.iloc[:, search]


class SurvivalEstimation(SurvivalCurves):
    """
    Instantiate class with data as survival curves. Columns must be timedelta objects
    """

    def __init__(self, survival_curves: pd.DataFrame, process_input_data=True):

        super().__init__(survival_curves)
        survival_curves = survival_curves.__deepcopy__()
        if process_input_data:
            survival_curves = process_survival_function(survival_curves)

        self.__survival = TimeCurveData(survival_curves)
        self.__check_args()

    def plot_residual_life(self, sample=None, mean_behaviour=True, unit=pd.to_timedelta(1, "D")):
        if not mean_behaviour:
            if sample is not None:
                residual_life_ = self.residual_life.sample(sample)
            else:
                residual_life_ = self.residual_life
            residual_life_.columns = pd.to_timedelta(self.survival_function.columns).total_seconds() / unit.total_seconds()
            color = plt.get_cmap("magma_r")(residual_life_.iloc[:, 0] / residual_life_.iloc[:, 0].max())

            residual_life_.T.plot(legend=False, ax=plt.gca(), color=color, lw=0.15, alpha=1)
            plt.grid()
            bounds = residual_life_.columns.min(), residual_life_.columns.max()
            plt.plot(bounds, bounds[::-1], c="k", alpha=0.5, lw=2, ls="--")
        else:
            des = (self.residual_life / unit).describe([0.05, .25, .5, .75, 0.95])
            des.columns = pd.to_timedelta(self.survival_function.columns).total_seconds() / unit.total_seconds()
            des = des.T.dropna().T
            des = des.drop(["count", "mean", "std"])
            for i, d in enumerate(des.index):
                c = plt.get_cmap("crest")(i / (len(des) - 1))
                des.loc[d].plot(color=c, ax=plt.gca())
                if i > 0:
                    plt.fill_between(
                        des.columns.to_list(),
                        des.iloc[i],
                        des.iloc[i - 1],
                        alpha=0.2,
                        facecolor=c)

            plt.grid()
            plt.legend()
            plt.ylabel("Expected residual lifespan")
            plt.xlabel("Time")

    def __check_args(self):
        if hasattr(self.survival_function.columns, "dt"):
            ValueError("curves index must have dt property")
