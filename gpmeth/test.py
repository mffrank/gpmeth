"""Functions to run tests"""
from typing import Optional, Tuple
from time import time
import numpy as np

from .util import RegressionData
from . import models as mod
import gpflow
import tensorflow_probability as tfp
import os
import json

# all_models_ps = (
#     mod.Constant(),
#     mod.ConstantLinear(),
#     mod.ConstantMatern(),
#     mod.RBFLinear(),
#     mod.RBFMatern(),
# )
# all_models_cat = (
#     mod.ConstantCategorical(),
#     mod.RBFCategorical(),
# )
#


def train_models(
    data: RegressionData,
    models: Tuple[mod.Model],
    null_model: Optional[mod.Model] = None,
    train_null: bool = True,
    num_inducing: int = 144,
    initialize_lengthscales=False,
    calculate_minmax=False,
    **kwargs,
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
                # null_model = gpflow.utilities.deepcopy(null_model)
                null_model.initialize_parameters_from_data(
                    data,
                    # compute_inducing_points=False,
                    inducing_dimensions="fixed",
                    num_points=num_inducing,
                    initialize_lengthscales=initialize_lengthscales,
                )
                # display(null_model)
                optres = null_model.optimize(data, initialize_parameters=False)
                print(f"Model {null_model._name} trained. Took {round(time() - t0)}s")
                print(f"Elbo: {optres.fun}")
                t0 = time()
                train_null = False
            # model = model.from_null(null_model)
            # model = gpflow.utilities.deepcopy(model)
            model.copy_null_parameters(null_model)
            #model.initialize_parameters_from_data(data, compute_inducing_points=False, **kwargs)
            # model.initialize_kernel_variances(0.001, only_trainable=True)
            # if(hasattr(model.kernel.kernels[1].kernels[0], "lengthscales")):
            #     # print(model.kernel.kernels[1].kernels[0].lengthscales.prior)
            #     import tensorflow_probability as tfp
            #     # model.kernel.kernels[1].kernels[0].lengthscales.prior = tfp.distributions.Gamma(6, 6/model.kernel.kernels[1].kernels[0].lengthscales.numpy())
        else:
            model.initialize_parameters_from_data(
                data,
                inducing_dimensions="fixed",
                num_points=num_inducing,
                initialize_lengthscales=initialize_lengthscales,
                **kwargs,
            )
            # if hasattr(model, "inducing_variable"):
            #     print(model.inducing_variable.Z)
        # display(model)
        to = time()
        optres = model.optimize(data, initialize_parameters=False)
        te = time()
        if calculate_minmax:
            mm = model.calculate_minmax()
            
        trained_models.append(model)
        print(f"Model {model._name} trained. Took {round(time() - t0, 2)}s ({round(te-to, 2)}s in optimizer, {round(to-t0, 2)} before, {round(time()-te, 2)} after)")
        print(f"Elbo: {optres.fun}")
    if null_model is not None:
        trained_models.append(null_model)

    return trained_models


# def serialize_dict(dictionary: dict, grp: str, file: str):
#     for k, v in dictionary.items():
#         dest = os.path.join(grp, k)
#         if isinstance(v, dict):
#             serialize_dict(v, dest, file)
#         print(v)
#         file[dest] = v


def save_models(
    model_list, outfile: str, path: Optional[str] = None, attrs: Optional[dict] = None, dtype=np.float64
):
    """Saves the models into an hdf5 store"""
    import h5py

    with h5py.File(outfile, mode="a") as f:
        for m in model_list:
            grp = os.path.join(path, type(m).__name__) if path else type(m).__name__

            # print(m.to_dict())
            if grp in f:
                del f[grp]

            # Write model parameters
            for k, v in m.to_dict().items():
                dest = os.path.join(grp, k)
                # f.create_dataset(dest, v, dtype=dtype)
                f[dest] = v
                
            # Write prior parameters
            for k, v in m.prior_dict().items():
                dest = os.path.join(grp, k)
                f[dest].attrs["prior_params"] = json.dumps(v.parameters)
                f[dest].attrs["prior_type"] = type(v).__name__
                
            # Write init parameters
            for k, v in m.init_params.items():
                f[grp].attrs[k] = v if v is not None else -1 # We save None as -1

        # Write region information
        if attrs is not None:
            for k, v in attrs.items():
                f[path].attrs[k] = v


def read_models(model_store, path):
    import h5py

    model_dict = dict()
    if isinstance(model_store, h5py.File):
        f = model_store
    else:
        f = h5py.File(model_store, "r")
    attrs = dict(f[path].attrs)
    models = list(f[path].keys())
    models = [x for x in models if x != 'ConstantLinear']
    # print(models)
    for model in models:
        dest = os.path.join(path, model)
        m = getattr(mod, model)
        init_params = {k:v if v != -1 else None for k,v in f[dest].attrs.items()} # We save None as -1
        model_dict[model] = m.from_dict(f[dest], **init_params)
        
        # Set the prior on parameters
        ref_dict = gpflow.utilities.parameter_dict(model_dict[model])
        for k, v in f[dest].items():
            if "prior_params" in v.attrs and "prior_type" in v.attrs:
                prior_params = json.loads(v.attrs["prior_params"])
                prior = getattr(tfp.distributions, v.attrs["prior_type"])(**prior_params)
                ref_dict[k].prior = prior
                
    if not isinstance(model_store, h5py.File):
        f.close()
    return model_dict, attrs


def test_models(model_store, path):
    model_dict, region_dict = read_models(model_store, path)
