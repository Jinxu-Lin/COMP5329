from Optimizers.sgd import SGD
from Optimizers.adam import Adam


def sgd(opt_params, args):
    return SGD(opt_params, args)

def adam(opt_params, args):
    return Adam(opt_params, args)
