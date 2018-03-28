from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class L1Loss(Operation):
    ''' Returns the L¹ loss Σ|xᵢ - yᵢ| averaged over the number of data points '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The model outputs for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N, any)
            The correct value for each of the N pieces of data.

        Returns
        -------
        The average L¹ loss.
        '''
        raise NotImplementedError


    def backward_a(self, grad):
        raise NotImplementedError


class MeanSquaredLoss(Operation):
    ''' Returns the mean squared error Σ(xᵢ - yᵢ)² over the data points '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The model outputs for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N, any)
            The correct value for each of the N pieces of data.

        Returns
        -------
        The mean squared error.
        '''
        raise NotImplementedError


    def backward_a(self, grad):
        raise NotImplementedError


class NegativeLogLikelihood(Operation):
    ''' Returns the negative log-likelihood. '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C log probabilities for each of the N pieces of data.
        
        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.
        
        Returns
        -------
        The average negative log-likelihood.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError

class KLDivergenceLoss(Operation):
    ''' Returns the Kullback-Leibler divergence loss from the outputs to the targets.
    
    The KL-Divergence loss for a single sample is given by yᵢ*(log(yᵢ) - xᵢ)
    '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The model outputs for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N, any)
            The correct vaue for each datum.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError

class CrossEntropyLoss(Operation):
    ''' Returns the cross-entropy loss between outputs and targets. '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.
       
        Returns
        -------
        The average cross-entropy loss.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError

class FocalLoss(Operation):
    ''' Returns the focal loss as described in https://arxiv.org/abs/1708.02002 

    Parameters
    ----------
    gamma : Real
        The focal factor.
    '''
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.

        Returns
        -------
        The average focal loss.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError
