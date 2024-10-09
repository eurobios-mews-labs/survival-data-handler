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


import pandas as pd
import numpy as np
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from survival_data_handler.main import SurvivalEstimation, Lifespan, TimeCurveData
from survival_data_handler.utils import smooth, process_survival_function, \
    compute_derivative
from survival_data_handler.base import TimeCurveInterpolation


@pytest.fixture()
def data():
    rossi = load_rossi()
    cph = CoxPHFitter()
    cph.fit(rossi, duration_col='week', event_col='arrest')
    curves = cph.predict_survival_function(rossi).T
    curves.columns = pd.to_timedelta(curves.columns.to_numpy() * 7, unit="D")
    return rossi, curves


def test_utils_smooth(data):
    _, curves = data
    ret = smooth(curves, freq="7D")
    assert all(abs(ret - curves) < 1)


def test_utils_process(data):
    _, curves = data
    ret = process_survival_function(curves)
    assert all(pd.DataFrame(ret == curves))
    curves.iloc[:, 1] = 1
    ret = process_survival_function(curves)
    assert any(pd.DataFrame(ret == curves))


def test_utils_compute_derivative(data):
    _, curves = data
    ret = compute_derivative(curves)
    assert all(pd.DataFrame(ret < 0))


def test_survival_estimation(data):
    _, curves = data
    se = SurvivalEstimation(
        curves.drop_duplicates())
    se.plot_residual_life()
    se.plot_residual_life(mean_behaviour=False)
    se.plot_residual_life(sample=10)


def test_survival_estimation_attributes(data):
    rossi, curves = data

    rossi["index"] = rossi.index
    se = SurvivalEstimation(survival_curves=curves)
    assert isinstance(se.hazard_function, TimeCurveData)
    assert isinstance(se.survival_function, TimeCurveData)
    assert isinstance(se.cumulative_hazard_function, TimeCurveData)


def test_lifespan(data):
    rossi, curves = data

    age = pd.to_timedelta(rossi["age"] * 365.25, unit="D")
    birth = pd.to_datetime('2000')
    curves = curves.iloc[:2]
    curves.index = ["a", "b"]

    rossi["index"] = np.random.choice(curves.index, replace=True, size=len(rossi))

    lifespan = Lifespan(
        curves,
        index=rossi["index"],
        birth=birth,
        age=age,
        window=(pd.to_datetime("2000"), pd.to_datetime("2100"))
    )

    # test plot function
    lifespan.plot_curves_residual_life()
    lifespan.add_supervision(event=rossi["arrest"],
                             durations=pd.to_timedelta(rossi["week"] * 7, unit="D"))

    assert isinstance(lifespan.survival_function, pd.DataFrame)
    assert isinstance(lifespan.residual_survival(pd.to_datetime("2022")), pd.DataFrame)
    assert isinstance(lifespan.percentile_life(0.1), pd.DataFrame)
    assert isinstance(lifespan.density_function, pd.DataFrame)
    assert isinstance(lifespan.cumulative_hazard_function, pd.DataFrame)
    assert isinstance(lifespan.hazard_function, pd.DataFrame)
    assert isinstance(lifespan.lifetime_distribution_function, pd.DataFrame)
    assert isinstance(lifespan.residual_life, pd.DataFrame)

    t = lifespan.compute_times(p=0.5)
    rossi["times"] = t


def test_supervision(data):
    from survival_data_handler import test_is_survival_curves
    rossi, curves = data
    rossi["duration"] = pd.to_timedelta(rossi["week"]*7, unit="D")
    age = pd.to_timedelta(rossi["age"] * 365.25, unit="D")
    birth = pd.to_datetime('2000')
    rossi["index"] = rossi.index
    lifespan = Lifespan(
        curves,
        index=rossi["index"],
        birth=birth,
        age=age,
        window=(pd.to_datetime("2000"), pd.to_datetime("2001"))
    )
    event = pd.Series(np.random.uniform(size=len(rossi)) > 0.5, index=rossi.index)
    lifespan.add_supervision(durations=rossi["duration"] + birth, event=event)

    lifespan.compute_confusion_matrix(on="survival_function", threshold=0.2)
    lifespan.compute_confusion_matrix(on="survival_function", threshold=0.2, score=True)
    lifespan.plot_average_tagged(on="survival_function")
    lifespan.plot_average_tagged(on="survival_function", plot_test_window=True, plot_type=None)
    lifespan.plot_dist_facet_grid(on="survival_function")
    lifespan.plot_tagged_sample(on="survival_function", n_sample=10)
    lifespan.plot_tagged_sample(on="survival_function")
    lifespan.plot_tagged_sample(on="survival_function", n_sample_pos=10)
    lifespan.plot_average_tagged(on="survival_function", event_type="censored")
    lifespan.plot_average_tagged(on="survival_function", event_type="observed")
    lifespan.plot_tagged_sample(on="hazard_function")
    test_is_survival_curves(lifespan.survival_function.round(3))
    lifespan.assess_metric(on="survival_function", method="roc-id")
    m = lifespan.assess_metric(on="survival_function", method="roc-cd")
    lifespan.assess_metric(on="survival_function", method="harrell")
    lifespan.plotly_roc_curve(on="survival_function")
    assert lifespan.supervised
    assert len(lifespan.remove_the_dead(on="survival_function")) <= len(lifespan.survival_function)

    lifespan.plotly_auc_vs_score("survival_function", m)


def test_curve_object(data):
    rossi, curves = data
    tc = TimeCurveData(curves)
    tc_prime = tc.derivative()
    assert isinstance(tc_prime, TimeCurveData)
    assert isinstance((-tc_prime), TimeCurveData)
    assert isinstance((tc_prime / tc), TimeCurveData)
    assert isinstance((tc_prime.exp() / tc), TimeCurveData)
    assert isinstance((tc_prime.log() - tc), TimeCurveData)
    assert isinstance((tc_prime.exp() + tc), TimeCurveData)


def test_interpolation_curves(data):
    rossi, curves = data
    rossi["duration"] = pd.to_timedelta(rossi["week"]*7, unit="D")
    rossi["birth"] = pd.to_datetime('2000')
    rossi["index"] = rossi.index
    curves = curves.iloc[:2]
    curves.index = ["a", "b"]

    rossi["index"] = np.random.choice(curves.index, replace=True, size=len(rossi))
    tc = TimeCurveData(curves)
    tc_prime = tc.interpolation

    ti = TimeCurveInterpolation(
        interpolation=tc_prime,
        index=rossi["index"],
        birth=rossi["birth"],
        period=pd.to_timedelta(30, "D"),
        window=(pd.to_datetime("2000"), pd.to_datetime("2001")))

    assert isinstance(ti.unique_curve, TimeCurveData)
    assert isinstance(ti.curve, TimeCurveData)
    assert len(ti.unique_curve) == 2
    assert len(ti.curve) == len(rossi)

    rossi, curves = data
    rossi["duration"] = pd.to_timedelta(rossi["week"]*7, unit="D")
    rossi["birth"] = pd.to_datetime('2000')
    rossi["index"] = rossi.index

    rossi["index"] = np.random.choice(curves.index, replace=True, size=len(rossi))
    tc = TimeCurveData(curves)
    tc_prime = tc.interpolation

    ti = TimeCurveInterpolation(
        interpolation=tc_prime,
        index=rossi["index"],
        birth=rossi["birth"],
        period=pd.to_timedelta(30, "D"),
        window=(pd.to_datetime("2000"), pd.to_datetime("2001")))

    assert isinstance(ti.unique_curve, TimeCurveData)
    assert isinstance(ti.curve, TimeCurveData)
    assert len(ti.unique_curve) <= len(rossi)
    assert len(ti.curve) == len(rossi)


def test_check_args(data):
    rossi, curves = data
    rossi["duration"] = pd.to_timedelta(rossi["week"]*7, unit="D")
    rossi["index"] = rossi.index
    index = rossi["index"]
    with pytest.raises(ValueError) as error:
        Lifespan(
            curves,
            index=index,
            birth=None,
            age=None,
            window=(pd.to_datetime("2000"), pd.to_datetime("2001"))
        )
    assert "age or birth" in str(error.value)
    assert index.name == rossi["index"].name
