"""Functions to run tests"""
from typing import Optional, Tuple
from time import time

from .util import RegressionData
from . import models as mod
import os

all_models_ps = (
    mod.Constant(),
    mod.ConstantLinear(),
    mod.ConstantMatern(),
    mod.RBFLinear(),
    mod.RBFMatern(),
)
all_models_cat = (
    mod.ConstantCategorical(),
    mod.RBFCategorical(),
)


def train_models(
    data: RegressionData,
    models: Tuple[mod.Model],
    null_model: Optional[mod.Model] = None,
    train_null: bool = True,
):
    # Train in 2 stages:
    # 1st stage: Train null models
    # if null_model is not None:
    #     null_model = null_model(fixed_genome_lengthscale=True)
    #     if train_null:
    #         t0 = time()
    #         null_model.optimize(data)
    #         print(f"Model {null_model._name} trained. Took {round(time() - t0)}s")

    # 2nd stage: Initialize full models with null models and train
    trained_models = []
    for model in models:
        t0 = time()
        if isinstance(model, mod.GPFullModel):
            if null_model is None:
                raise ValueError(
                    "Must specify null model when training a model of the class GPFullModel"
                )
            if train_null:
                t0 = time()
                null_model.initialize_parameters_from_data(
                    data,
                    inducing_dimensions="fixed",
                    num_points=144,
                )
                null_model.optimize(data, initialize_parameters=False)
                print(f"Model {null_model._name} trained. Took {round(time() - t0)}s")
                train_null = False
            # model = model.from_null(null_model)
            model.copy_null_parameters(null_model)
            model.initialize_parameters_from_data(data, compute_inducing_points=False)
        else:
            model.initialize_parameters_from_data(
                data,
                inducing_dimensions="fixed",
                num_points=144,
            )
            # if hasattr(model, "inducing_variable"):
            #     print(model.inducing_variable.Z)
        # print(model.to_dict())
        model.optimize(data, initialize_parameters=False)
        trained_models.append(model)
        print(f"Model {model._name} trained. Took {round(time() - t0)}s")

    if null_model is not None:
        trained_models.append(null_model)

    return trained_models


def save_models(
    model_list, outfile: str, path: Optional[str] = None, attrs: Optional[dict] = None
):
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
        if attrs is not None:
            for k, v in attrs.items():
                f[path].attrs[k] = v


def read_models(model_store, path):
    import h5py

    model_dict = dict()
    with h5py.File(model_store, "r") as f:
        attrs = dict(f[path].attrs)
        models = list(f[path].keys())
        for model in models:
            dest = os.path.join(path, model)
            m = getattr(mod, model)
            model_dict[model] = m.from_dict(f[dest])
    return model_dict, attrs


def test_models(model_store, path):
    model_dict, region_dict = read_models(model_store, path)
