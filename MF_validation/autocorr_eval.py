import numpy as np
from scipy.optimize import minimize
from scipy import integrate


### Yann Zerlaut codes for ACF -----------------------------------------------------
### --------------------------------------------------------------------------------
def autocorrel(Signal, tmax, dt):
    """
    YANN ZERLAUT COURTESY
    argument : Signal (np.array), tmax and dt (float)
    tmax, is the maximum length of the autocorrelation that we want to see
    returns : autocorrel (np.array), time_shift (np.array)
    take a Signal of time sampling dt, and returns its autocorrelation
     function between [0,tstop] (normalized) !!
    """
    steps = int(tmax/dt) # number of steps to sum on
    Signal2 = (Signal-Signal.mean())/Signal.std()
    cr = np.correlate(Signal2[steps:],Signal2)/steps
    time_shift = np.arange(len(cr))*dt

    return cr/cr.max(), time_shift

def get_acf_time(Signal, dt, acf,
                 min_time=1., max_time=100.,
                 procedure='integrate'):
    #if acf is non devo passare min_time = 0 e max_time = 1 --> dominio del mio segnale
    """
    YANN ZERLAUT COURTESY
    returns the autocorrelation time of some fluctuations: Signal
    two methods: fitting of an exponential decay vs temporal integration
    """

    if acf is None:
        acf, shift = autocorrel(Signal, np.mean([max_time, min_time]), dt)

    if procedure=='fit':
        def func(X):
            return np.sum(np.abs(np.exp(-shift/X[0])-acf))
        res = minimize(func, [min_time],
                       bounds=[[min_time, max_time]], method='L-BFGS-B')
        return res.x[0]
    else:
        # we integrate
        shift = np.arange(len(acf))*dt
        return integrate.cumtrapz(acf, shift)[-1]
### --------------------------------------------------------------------------------
### --------------------------------------------------------------------------------