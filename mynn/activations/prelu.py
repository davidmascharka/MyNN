import numpy as np

from mygrad import Tensor
from mygrad.operation_base = Operation

__all__  ['prelu']

class prelu:
    ''' Returns the parametric rectified linear activation elementwise along x. The PReLU is given
    by max(x, 0) + slope*min(x, 0), where `slope` is learnable.
    '''
    def __init__(self, slope):
        '''
        '''
        pass
