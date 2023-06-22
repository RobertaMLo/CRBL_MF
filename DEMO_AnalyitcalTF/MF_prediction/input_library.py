import numpy as np
from scipy import signal


def gauss_inp(t_len, std, noise_freq, freq, minval):
    y = signal.gaussian(t_len, std) + np.random.rand(t_len)*noise_freq
    y = (y*freq)+minval
    return y


def syn_input_absval(time, sig_freq, minval, ampl):

    y = np.abs(np.sin(sig_freq * time)) * ampl
    y = y + minval
    return y


def syn_input(time, sig_freq, minval, ampl):

    y = np.sin(sig_freq * time) * ampl
    y = y + minval
    return y


def rect_input(time, t_start, t_end, minval, freq, noise_freq):

    """
    time = time vector of simulation
    t_start = start of the step INDEX
    t_end = end of the step INDEX
    minval = baseline value (deviation from 0)
    freq = peak value
    noise_freq = random noise frequencies
    """

    y = np.ones(len(time)) * freq + np.random.rand(len(time)) * noise_freq
    y[:t_start] = y[:t_start]*0+np.random.rand(t_start)*noise_freq
    y[t_end:] = y[t_end:]*0+np.random.rand(len(time) - t_end)*noise_freq
    y = y + minval

    return y

def impulse_train(time, pulse_period, pulse_width, pulse_max_freq, pulse_baseline, noise_freq):
    #pulse period = quanti ne voglio, pulse_width = quanto larghi
    y = np.arange(len(time)-100) % pulse_period < pulse_width #se il resto della divisione è minore di D
    #n_imp = time.max()/pulse_width
    y = y*pulse_max_freq
    v1 = np.zeros(100)
    y = np.concatenate((v1, y), axis=None)
    y = (y + np.random.rand(len(time))*noise_freq) + pulse_baseline
    return y


def impulse_train_s500(time, pulse_period, pulse_width, pulse_max_freq, pulse_baseline, noise_freq):
    #pulse period = quanti ne voglio, pulse_width = quanto larghi
    y = np.arange(len(time)-500) % pulse_period < pulse_width #se il resto della divisione è minore di D
    #n_imp = time.max()/pulse_width
    y = y*pulse_max_freq
    v1 = np.zeros(500)
    y = np.concatenate((v1, y), axis=None)
    y = (y + np.random.rand(len(time))*noise_freq) + pulse_baseline
    return y


def EBCC_protocol(time, tstartCS, tstartUS, tendCS, tendUS, minval, freqCS, freqUS, noise_freq):
    CS = rect_input(time, tstartCS, tendCS, minval, freqCS, noise_freq)
    US = rect_input(time, tstartUS, tendUS, minval, freqUS, noise_freq)
    return CS, US
