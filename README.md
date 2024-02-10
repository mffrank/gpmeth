# GPmeth

#
GPmeth is a modeling framework based on [GPflow](https://github.com/GPflow). It detects differential DNA methylation / chromatin accessibility in single-cell multimodal datasets that study continuous biological processes. The idea is to generate temporal trajectories based on one modality (typically RNA-seq) and then model methylation/accessibility along these continuous axes with a Gaussian process model. GPmeth models methylation at base resolution and can automatically refine boundaries of differentially methylated/accessible regions.

# Installation

GPmeth is built with [GPflow](https://github.com/GPflow), which dependso on [TensorFlow](https://www.tensorflow.org/) and [TensorFlow Probability](https://www.tensorflow.org/probability). I recommend installing these first.

```
# Install tensorflow and tensorflow-probability
pip install gpflow tensorflow~=2.10.0 tensorflow-probability~=0.18.0

# Install gpmeth
git clone git@github.com:mffrank/gpmeth.git
cd gpmeth
pip install .

```
