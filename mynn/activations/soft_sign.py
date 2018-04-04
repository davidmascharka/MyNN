from mygrad import abs

def soft_sign(x):
    ''' Returns the soft sign function x/(1 + |x|) '''
    return x / (1 + abs(x))
