import numpy as np
import sys
sys.path.append('../')
from CRBL_MF_Model.MF_prediction import *
from CRBL_MF_Model.MF_prediction.load_config_TF import *
from CRBL_MF_Model.MF_prediction.input_library import *
from scipy.integrate import odeint
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def build_up_differential_operator_first_order(TF1, TF2, w, T):
    """
    simple first order system
    """

    # exc aff = fmossy

    def A0(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return 1./T*(TF1(exc_aff+pure_exc_aff, V[1]+inh_aff, w)-V[0])
    
    def A1(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return 1./T*(TF2(exc_aff, V[0], V[1]+inh_aff, w)-V[1])
    
    def Diff_OP(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
    return Diff_OP
    

def build_up_differential_operator_mossy(TF1, TF2, w, \
                                   Ngrc, Ngoc, Nmossy, T):
    """
    Implements Equation (3.16) in El BOustani & Destexhe 2009
    in the case of a network_scaffold of two populations:
    one excitatory and one inhibitory

    Each neuronal population has the same transfer function
    this 2 order formalism computes the impact of finite size effects
    T : is the bin for the Markovian formalism to apply

    the time dependent vector vector is V=[Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm]
    the function returns Diff_OP
    and d(V)/dt = Diff_OP(V)
    """

    # dVgrc/dt : grc activity
    def A0(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                    .5 * V[7] * diff2_fe_fe(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    .5 * V[5] * diff2_fe_fi(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    .5 * V[5] * diff2_fi_fe(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    .5 * V[6] * diff2_fi_fi(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    TF1(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0])

    # dVgoc/dt : goc activity
    def A1(V, inh_aff=0, pure_exc_aff=0):
        #print('==========================', diff2_fgoc_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff))
        return 1. / T * (
                    .5 * V[2] * diff2_fgrc_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[7] * diff2_fm_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[4] * diff2_fgrc_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[3] * diff2_fgrc_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[3] * diff2_fgoc_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[5] * diff2_fgoc_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .4 * V[4] * diff2_fm_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[5] * diff2_fm_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    .5 * V[6] * diff2_fgoc_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    TF2(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1])

    # dCgrcgrc/dt : grc variance
    def A2(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                    1. / Ngrc * TF1(V[8] + pure_exc_aff, V[1] + inh_aff, w) * (
                        1. / T - TF1(V[8] + pure_exc_aff, V[1] + inh_aff, w)) +
                    (TF1(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) ** 2 +
                    2. * V[4] * diff_fe(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    2. * V[3] * diff_fi(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    -2. * V[2])

    # dCgrcgoc/dt : grc and goc covariance
    # N.B.: dCgocgrc/dt = dCgrcgoc/dt
    def A3(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                    (TF1(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) * (
                        TF2(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) +
                    V[2] * diff_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    V[3] * diff_fe(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    V[3] * diff_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    V[6] * diff_fi(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    V[5] * diff_fe(TF1, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    V[4] * diff_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                    -2. * V[3])

    # dCgrcm/dt : grc and mossy covariance
    def A4(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        return 1 / T *(
                    V[5] * diff_fi(TF1, V[8] + pure_exc_aff, V[1] + inh_aff) +
                    V[7] * diff_fe(TF1, V[8] + pure_exc_aff, V[1] + inh_aff)
                    - 2. * V[4])

    # dCgocm/dt : goc and mossy covariance
    def A5(V, inh_aff=0, pure_exc_aff=0):
        return 1/ T *(
                  V[4] * diff_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                  V[5] * diff_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                  V[7] * diff_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                  -2. * V[5])

    # dCgocgoc/dt : goc variance
    def A6(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                    1. / Ngoc * TF2(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) * (
                        1. / T - TF2(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w)) +
                    (TF2(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) ** 2 +
                    2. * V[3] * diff_fgrc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    2. * V[6] * diff_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                    2. * V[5] * diff_fm_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                    -2. * V[6])

    # dCmm/dt : mossy variance
    def A7(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        return 1. / T * (
                    1. / Nmossy * (V[8] * (
                        1. / T - V[8]))
                    -2. * V[7])

    def A8(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        #return 1. /T * ((V[8]+1 - V[8])/1)
        return V[8]

    def Diff_OP(V, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A1(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A2(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A3(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A4(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A5(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A6(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A7(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A8(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)], dtype=object)

    return Diff_OP

##### Derivatives taken numerically,
## to be implemented analitically ! not hard...

# TF differentials for GrC and all pops with two input connections -----------------------------------------------------
def diff_fe(TF, fe, fi, w=0, df=1e-4):
    #print('TF +DF', TF(fe+df_mf/2., fi, w))
    #print('TF-DF', TF(fe-df_mf/2.,fi, w))
    return (TF(fe+df/2., fi, w)-TF(fe-df/2.,fi, w))/df

def diff_fi(TF, fe, fi, w=0, df=1e-4):
    return (TF(fe, fi+df/2., w)-TF(fe, fi-df/2., w))/df

def diff2_fe_fe(TF, fe, fi, w=0, df=1e-4):
    return (diff_fe(TF, fe+df/2., fi, w)-diff_fe(TF, fe-df/2., fi, w))/df

def diff2_fi_fe(TF, fe, fi, w=0, df=1e-4):
    return (diff_fi(TF, fe+df/2., fi, w)-diff_fi(TF,fe-df/2.,fi, w))/df

def diff2_fe_fi(TF, fe, fi, w=0, df=1e-4):
    return (diff_fe(TF, fe, fi+df/2., w)-diff_fe(TF,fe, fi-df/2., w))/df

def diff2_fi_fi(TF, fe, fi, w=0, df=1e-4):
    return (diff_fi(TF, fe, fi+df/2., w)-diff_fi(TF,fe, fi-df/2., w))/df


# TF differentials for Golgi cells -------------------------------------------------------------------------------------
def diff_fm_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (TF_goc(fm+df/2., fgrc, fgoc, w)-TF_goc(fm-df/2., fgrc, fgoc, w))/df

def diff_fgrc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (TF_goc(fm, fgrc+df/2., fgoc, w)-TF_goc(fm, fgrc-df/2., fgoc, w))/df

def diff_fgoc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (TF_goc(fm, fgrc, fgoc+df/2., w)-TF_goc(fm, fgrc, fgoc-df/2., w))/df

def diff2_fm_fm_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fm_goc(TF_goc, fm+df/2., fgrc, fgoc, w)-diff_fm_goc(TF_goc, fm-df/2., fgrc, fgoc, w))/df

def diff2_fgrc_fgrc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fgrc_goc(TF_goc, fm, fgrc+df/2., fgoc, w)-diff_fgrc_goc(TF_goc,fm, fgrc-df/2., fgoc, w))/df

def diff2_fgoc_fgoc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fgoc_goc(TF_goc, fm, fgrc, fgoc+df/2., w)-diff_fgoc_goc(TF_goc,fm, fgrc, fgoc-df/2., w))/df

def diff2_fm_fgrc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fm_goc(TF_goc, fm, fgrc+df/2., fgoc, w)-diff_fm_goc(TF_goc, fm, fgrc-df/2., fgoc, w))/df

def diff2_fgrc_fm_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
   return (diff_fgrc_goc(TF_goc, fm+df/2., fgrc, fgoc, w)-diff_fgrc_goc(TF_goc, fm-df/2., fgrc, fgoc, w))/df

def diff2_fm_fgoc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fm_goc(TF_goc, fm, fgrc, fgoc+df/2., w)-diff_fm_goc(TF_goc, fm, fgrc, fgoc-df/2., w))/df

def diff2_fgoc_fm_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fgoc_goc(TF_goc, fm+df/2., fgrc, fgoc, w)-diff_fgoc_goc(TF_goc, fm-df/2., fgrc, fgoc, w))/df

def diff2_fgrc_fgoc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fgrc_goc(TF_goc, fm, fgrc, fgoc+df/2., w)-diff_fgrc_goc(TF_goc, fm, fgrc, fgoc-df/2., w))/df

def diff2_fgoc_fgrc_goc(TF_goc, fm, fgrc, fgoc, w=0, df=1e-4):
    return (diff_fgoc_goc(TF_goc, fm, fgrc+df/2., fgoc, w)-diff_fgoc_goc(TF_goc, fm, fgrc-df/2., fgoc, w))/df


def find_fixed_point_first_order(NRN1, NRN2, NTWK, FILE_GrC, FILE_GoC, CI_vec1, t, w, exc_aff, T,
                                 verbose=True):


    TF1 = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=2)
    TF2 = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.2)

    ### FIRST ORDER ##

    X = CI_vec1
    X_vett = np.zeros((len(t), 2))
    for i in range(len(t)):
        last_X = X
        X = X + (t[1] - t[0]) * build_up_differential_operator_first_order(TF1, TF2, w, T)(X, exc_aff[i])
        X_vett[i, :] = X


    if verbose:
        print('first order prediction: ', X_vett[-1])
    return X_vett, X_vett[-1][0], X_vett[-1][1]


def find_fixed_point_mossy(NRN1, NRN2, NTWK, FILE_GrC, FILE_GoC, CI_vec2, t, w, fmossy, Ngrc, Ngoc, Nmossy, T,
                           verbose=False):

    #[Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm]

    TF1 = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=2)
    TF2 = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.2)
    ### SECOND ORDER ###

    # simple euler

    X = CI_vec2
    X_vett = np.zeros((len(t), 9))
    for i in range(len(t)):
        last_X = X
        X = X + (t[1] - t[0]) * build_up_differential_operator_mossy(TF1, TF2, w, Ngrc=Ngrc, Ngoc=Ngoc, Nmossy=Nmossy,
                                                                     T=T)(X)
        #X[8] = fmossy[i]/T
        X[8] = fmossy[i]
        #X[4], X[5], X[7] = 0, 0, 0
        X_vett[i, :] = X

    print('Make sure that those two values are similar !!')
    print('X: ', X)
    print('last X: ', last_X)

    if verbose:
        print(X)
    if verbose:
        print('first order prediction: ', X[-1])

    # return X[-1][0], X[-1][1], np.sqrt(X[-1][2]), np.sqrt(X[-1][3]), np.sqrt(X[-1][4])
    # return X, X[0], X[1], np.sqrt(X[2]), np.sqrt(X[3]), np.sqrt(X[4])
    #return X_vett, X, X[0], X[1], np.sqrt(X[2]), np.sqrt(X[3]), np.sqrt(X[4])
    return X_vett


def plot_for_thesis_2order(t, X, F_mossy, font_size = 8, linew=1.5):

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize =(7,5))
    fig.suptitle('Second Order GRL-MF: Mean activity and Covariances [Hz]', fontsize = font_size +2)

    # GrC --------------------------------------------------------------------------------------------------------------
    ax1.plot(t, X[:,0], '#62DACF', linewidth = linew)
    ax1.set_ylabel('$\\nu_{GrC}$', fontsize = font_size + 1)
    ax1.set_xticks([])

    # GoC --------------------------------------------------------------------------------------------------------------
    ax2.plot(t, X[:,1], '#7E8ACA', linewidth = linew)
    ax2.set_ylabel('$\\nu_{GoC}$', fontsize = font_size+1)
    ax2.set_xticks([])

    # Var GrC --------------------------------------------------------------------------------------------------------------
    ax3.plot(t, X[:,2], 'm', linewidth = linew)
    ax3.set_ylabel('$C_{GoC}$', fontsize = font_size+1)
    ax3.set_xticks([])

    # Covar GrCGoC --------------------------------------------------------------------------------------------------------------
    ax4.plot(t, X[:,3], 'm', linewidth = linew)
    ax4.set_ylabel('$C_{GrCGoC}$', fontsize = font_size+1)
    ax4.set_xticks([])

    # Var GoC --------------------------------------------------------------------------------------------------------------
    ax5.plot(t, X[:,4], 'm', linewidth = linew)
    ax5.set_ylabel('$C_{GoC}$', fontsize = font_size+1)
    ax5.set_xticks([])

    # Mossy ------------------------------------------------------------------------------------------------------------
    ax6.plot(t, F_mossy, '#D11D12', linewidth = linew)
    ax6.set_ylabel('$\\nu_{drive}$', fontsize = font_size+1)
    ax6.set_xlabel('time [s]', fontsize = font_size+1)

    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.15, top=0.92, bottom = 0.1)
    plt.show()

def plot_for_thesis_2order_activity_mossy(t, X, F_mossy, font_size = 8, linew=1.5):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize =(7,5))
    fig.suptitle('Second Order GRL-MF: Mean activity [Hz]', fontsize = font_size +2)

    # GrC --------------------------------------------------------------------------------------------------------------
    ax1.plot(t, X[:,0], '#62DACF', linewidth = linew)
    ax1.set_ylabel('$\\nu_{GrC}$', fontsize = font_size + 1)
    ax1.set_xticks([])

    # GoC --------------------------------------------------------------------------------------------------------------
    ax2.plot(t, X[:,1], '#7E8ACA', linewidth = linew)
    ax2.set_ylabel('$\\nu_{GoC}$', fontsize = font_size+1)
    ax2.set_xticks([])

    # Mossy ------------------------------------------------------------------------------------------------------------
    ax3.plot(t, F_mossy, '#D11D12', linewidth = linew)
    ax3.set_ylabel('$\\nu_{drive}$', fontsize = font_size+1)
    ax3.set_xlabel('time [s]', fontsize = font_size+1)

    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.15, top=0.92, bottom = 0.1)
    plt.show()

def plot_for_thesis_2order_variance_mossy(t, X, font_size = 8, linew=1.5):

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize =(7,5))
    fig.suptitle('Second Order GRL-MF: Variance [Hz]', fontsize = font_size +2)

    #[Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm]
    # Cgrcgrc -----------------------------------------------------------------------------------------------------------
    ax1.plot(t, X[:,2], '#62DACF', linewidth = linew)
    ax1.set_ylabel('$\\sigma_{GrC}$', fontsize = font_size + 1)
    ax1.set_xticks([])

    # Cgrcgoc ----------------------------------------------------------------------------------------------------------
    ax2.plot(t, X[:,3], 'b', linewidth = linew)
    ax2.set_ylabel('$\\sigma_{GrCGoC}$', fontsize = font_size+1)
    ax2.set_xticks([])

    # Cgrcm ------------------------------------------------------------------------------------------------------------
    ax3.plot(t, X[:,4], 'rebeccapurple', linewidth = linew)
    ax3.set_ylabel('$\\sigma_{GrCMossy}$', fontsize = font_size+1)
    ax3.set_xlabel('time [s]', fontsize = font_size+1)

    # Cmgoc ------------------------------------------------------------------------------------------------------------
    ax4.plot(t, X[:,5], 'm', linewidth = linew)
    ax4.set_ylabel('$\\sigma_{MossyGoC}$', fontsize = font_size+1)
    ax4.set_xlabel('time [s]', fontsize = font_size+1)

    # Cgocgoc ----------------------------------------------------------------------------------------------------------
    ax5.plot(t, X[:,6], '#7E8ACA', linewidth = linew)
    ax5.set_ylabel('$\\sigma_{GoC}$', fontsize = font_size+1)
    ax5.set_xlabel('time [s]', fontsize = font_size+1)

    # Cmm ----------------------------------------------------------------------------------------------------------
    ax6.plot(t, X[:,7], 'crimson', linewidth = linew)
    ax6.set_ylabel('$\\sigma_{Mossy}$', fontsize = font_size+1)
    ax6.set_xlabel('time [s]', fontsize = font_size+1)

    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.15, top=0.92, bottom = 0.1)
    plt.show()

def plot_for_thesis_1vs2order(t, X1, X, F_mossy, font_size = 8, linew=1.5):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5))
    fig.suptitle('First vs. Second Order GRL-MF: Mean activity [Hz]', fontsize=font_size + 2)

    # GrC --------------------------------------------------------------------------------------------------------------
    ax1.plot(t, X1[:, 0], 'darkcyan', linewidth=linew)
    ax1.plot(t, X[:, 0], '#62DACF', linewidth=linew)
    ax1.set_ylabel('$\\nu_{GrC}$', fontsize=font_size + 1)
    ax1.legend(['First Order', 'Second Order'])
    ax1.set_xticks([])

    # GoC --------------------------------------------------------------------------------------------------------------
    ax2.plot(t, X1[:, 1], 'navy', linewidth=linew)
    ax2.plot(t, X[:, 1], '#7E8ACA', linewidth=linew)
    ax2.set_ylabel('$\\nu_{GoC}$', fontsize=font_size + 1)
    ax2.legend(['First Order', 'Second Order'])
    ax2.set_xticks([])

    # Mossy ------------------------------------------------------------------------------------------------------------
    ax3.plot(t, F_mossy, '#D11D12', linewidth=linew)
    ax3.set_ylabel('$\\nu_{drive}$', fontsize=font_size + 1)
    ax3.set_xlabel('time [s]', fontsize=font_size + 1)

    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, top=0.92, bottom=0.1)
    plt.show()


if __name__=='__main__':

    # find_fixed_point('LIF', 'LIF', 'Vogels-Abbott', exc_aff=0., Ne=4000, Ni=1000, verbose=True)
    FILE_GrC = '/home/bcc/bsb-ws/CRBL_MF_Model/20211019_110203_GrC_CRBL_CONFIG_tsim5_alpha2.0_fit.npy'
    FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/20211014_222536_GoC_CRBL_CONFIG_tsim5_alpha1.2_fit.npy'

    #FILE_GrC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20211019_110203_GrC_CRBL_CONFIG_tsim5_alpha2.0_fit.npy'
    #FILE_GoC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20211014_222536_GoC_CRBL_CONFIG_tsim5_alpha1.2_fit.npy'
    TF1 = load_transfer_functions('GrC', 'CRBL_CONFIG', FILE_GrC, alpha=2)
    TF2 = load_transfer_functions_goc('GoC', 'CRBL_CONFIG', FILE_GoC, alpha=1.2)

    w = 0
    t = np.arange(5000) * 1e-4
    T = 3.5e-3 #5e-3

    #fmossy = syn_input(time=t,sig_freq=200, minval=30,ampl=40) + np.random.rand(len(t))*30

    #fmossy = rect_input(time=t, t_start=1000, t_end=3000, minval=0, freq=50, noise_freq=4)

    #fmossy = np.random.rand(len(time))*50

    #fmossy = gauss_inp(t_len=len(t), std=20, noise_freq=0.1, freq=50, minval=0)

    fmossy = impulse_train(time=t, pulse_period=100, pulse_width=5, pulse_max_freq = 50, pulse_baseline=0, noise_freq=4)

    CI_vec1 = [0.5, 10]
    #CI_vec2 = [0.5, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0]]


    ## First order check
    X1, fix_grc, fix_goc = find_fixed_point_first_order('GrC', 'GoC', 'CRBL_CONFIG', FILE_GrC, FILE_GoC, CI_vec1, t, w,
                                                        fmossy, T)

    #plot_for_thesis_2order_activity_mossy(time_plot, X1, fmossy_plot)

    #X, grc_fix, goc_fix, vgrc_fix, covgrcgoc_fix, vgoc_fix = find_fixed_point('GrC', 'GoC', 'CRBL_CONFIG', FILE_GrC, FILE_GoC, w, fmossy, Ne=8000, Ni=2000, verbose=True)

    #X, last_X, grc_fix, goc_fix, vgrc_fix, covgrcgoc_fix, vgoc_fix = find_fixed_point('GrC', 'GoC', 'CRBL_CONFIG', FILE_GrC,
    #                                                                          FILE_GoC, w, fmossy, Ne=8000, Ni=2000,
    #                                                                          verbose=True)

    Ngrc = 28615 #8000 #28615
    Ngoc = 70 #2000 #70
    Nmossy = 2336 #2000 #2336

    #CI_vec2 = [0.5, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,fmossy[0]]
    # we start from the first order prediction !!!
    CI_vec2 = [fix_grc, fix_goc, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0]]

    X = find_fixed_point_mossy('GrC', 'GoC', 'CRBL_CONFIG', FILE_GrC, FILE_GoC, CI_vec2, t, w, fmossy, Ngrc, Ngoc,
                               Nmossy, T, verbose=False)


    plot_for_thesis_1vs2order(t, X1, X, fmossy)

    plot_for_thesis_2order_activity_mossy(t, X, fmossy)
    plot_for_thesis_2order_variance_mossy(t, X)
