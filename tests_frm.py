#
import numpy as np
import pytest

import candle_improve_utils as utils


@pytest.mark.parametrize("metrics", [["mse"], ["mse", "pcc"], ["mse", "rmse", "pcc", "scc", "r2"]])
def test_metrics_func(metrics):
    y = np.random.randn(10)
    f = np.random.randn(10)

    try:
        meval = utils.compute_metrics(y, f, metrics)
    except Exception as e:
        print(e)
        assert 0
    else:
        mseval = utils.mse(y, f)
        rmseval = utils.rmse(y, f)
        pccval = utils.pearson(y, f)
        sccval = utils.spearman(y, f)
        r2val = utils.r_square(y, f)

        assert np.allclose(mseval, meval["mse"])
        if "pcc" in metrics:
            assert np.allclose(pccval, meval["pcc"])
        if "scc" in metrics:
            assert np.allclose(sccval, meval["scc"])
        if "rmse" in metrics:
            assert np.allclose(rmseval, meval["rmse"])
        if "r2" in metrics:
            assert np.allclose(r2val, meval["r2"])

