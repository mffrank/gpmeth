import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Probit(tfp.bijectors.Bijector):
    """Bijector class for the probit transformation"""

    def __init__(self, validate_args=False, name="probit"):
        super(Probit, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name
        )

    def _forward(self, x):
        return np.sqrt(2.0) * tf.math.erfinv(2 * x - 1)

    def _inverse(self, y):
        jitter = 1e-3  # ensures output is strictly between 0 and 1
        return 0.5 * (1.0 + tf.math.erf(y / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
