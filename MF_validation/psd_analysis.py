from CRBL_MF_Model.MF_validation.predictions_analysis import compute_PSD
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse


def load_mf_signal(FILENAME):

    data_mf = np.load(FILENAME)

    return data_mf[0], data_mf[1], data_mf[2], data_mf[3], data_mf[4]

def compute_fft(ybias):

    dt = 1e-4
    fs = 1/dt
    t = np.arange(0, len(ybias)*dt, dt)


    y = ybias-np.mean(ybias)
    y_fft = np.fft.rfft(y)               # Original DFT computed with FFt algorithm
    #y_fft = y_fft[:round(len(t)/2)]     # First half ONLY FOR REAL INPUT. FFT of real input symmetric around 0
    y_fft = np.abs(y_fft)               # Absolute value of magnitudes (without abs some freqs are negative)
    y_fft = y_fft/max(y_fft)            # Normalized on the maximum fft value so max = 1


    fig, ax = plt.subplots(1, 2)
    ax = ax.flatten()
    ax[0].plot(t*1e3, ybias)
    ax[0].set_title("Original signal")
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("Activity [Hz]")
    freq_x_axis = np.linspace(0, fs/2, len(y_fft))
    ax[1].plot(freq_x_axis, y_fft, "o-")
    ax[1].set_title("Frequency magnitudes")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

    f_loc = np.argmax(y_fft) # Finds the index of the max
    f_val = freq_x_axis[f_loc] # The strongest frequency value
    print(f"The strongest frequency is f = {f_val}")

    return y_fft



if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Compute PSD'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-FILENAME',
                        help="filename where the scores are saved")

    args = parser.parse_args()

    GrC, GoC, MLI, PC, mf = load_mf_signal(args.FILENAME)

    compute_fft(GrC)
