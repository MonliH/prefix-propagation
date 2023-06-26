from gpytorch.kernels import (
    RBFKernel,
    PolynomialKernel,
)

from gpytorch.constraints import Interval

import math


def get_kernel(kernel_key, key_dim, out_scale=None, out_batch_shape=None):
    try:
        if kernel_key == "RBF":
            kernel = RBFKernel()
            kernel.lengthscale = 64
            kernel.raw_lengthscale.requires_grad = False
        elif kernel_key == "Poly":
            kernel = PolynomialKernel(power=2)
        else:
            raise NotImplementedError


        return kernel

    except Exception as e:
        print(e)

