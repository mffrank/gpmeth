"""Functions to run tests"""
from typing import Optional, Tuple

from gpflow.models.training_mixins import RegressionData
from . import models as mod
import os

all_pseudotime_models = (
    mod.Constant,
    mod.ConstantMatern,
    mod.RBFMatern,
)


def train_models_pseudotime(
    data: RegressionData,
    models: Tuple[mod.Model] = all_pseudotime_models,
    null_model=mod.RBFConstant,
):
    # Train in 2 stages:
    # 1st stage: Train null models
    null_model = null_model(fixed_genome_lengthscale=True)
    null_model.optimize(data)

    # 2nd stage: Initialize full models with null models and train
    trained_models = []
    for model in models:
        if isinstance(model, mod.GPFullModel):
            model = model.from_null(null_model)
        else:
            model = model()
        model.optimize(data)
        trained_models.append(model)

    trained_models.append(null_model)

    return trained_models


def save_models(model_list, outfile: str, path: Optional[str] = None):
    """Saves the models into an hdf5 store"""
    import h5py

    with h5py.File(outfile, "a") as f:
        for m in model_list:
            grp = os.path.join(path, type(m).__name__)
            if grp in f:
                del f[grp]
            for k, v in m.to_dict().items():
                dest = os.path.join(grp, k)
                f[dest] = v


def read_models(model_store, path):
    import h5py

    model_list = []
    with h5py.File(model_store, "r") as f:
        models = list(f[path].keys())
        for model in models:
            dest = os.path.join(path, model)
            model = getattr(mod, model)
            model_list.append(model.from_dict(f[dest]))
    return model_list
