#
import numpy as np
import pytest

import candle_improve_utils as utils


@pytest.mark.parametrize("metrics", [["mse"], ["mse", "pcc"], ["mse", "rmse", "pcc", "scc", "r2"]])
def test_metrics_func(metrics):
    y_true = np.random.randn(10)
    y_pred = np.random.randn(10)

    try:
        meval = utils.compute_metrics(y_true, y_pred, metrics)
    except Exception as e:
        print(e)
        assert 0
    else:
        mseval = utils.mse(y_true, y_pred)
        rmseval = utils.rmse(y_true, y_pred)
        pccval = utils.pearson(y_true, y_pred)
        sccval = utils.spearman(y_true, y_pred)
        r2val = utils.r_square(y_true, y_pred)

        assert np.allclose(mseval, meval["mse"])
        if "pcc" in metrics:
            assert np.allclose(pccval, meval["pcc"])
        if "scc" in metrics:
            assert np.allclose(sccval, meval["scc"])
        if "rmse" in metrics:
            assert np.allclose(rmseval, meval["rmse"])
        if "r2" in metrics:
            assert np.allclose(r2val, meval["r2"])

