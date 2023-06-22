import numpy as np
import sys
sys.path.append('../')
from MF_prediction import *
from MF_prediction.load_config_TF import *
from MF_prediction.input_library import *
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from MF_validation.plot_babe import *
import time

def build_up_differential_operator_first_order(TF1, TF2, TF3, TF4, w, T):
    """
    simple first order system
    """

    # exc aff = fmossy

    def A0(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF1(exc_aff+pure_exc_aff, V[1]+inh_aff, w)-V[0])
    
    def A1(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF2(exc_aff+pure_exc_aff, V[0], V[1]+inh_aff, w)-V[1])

    def A2(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF3(V[0]+pure_exc_aff, V[2]+inh_aff,  w)-V[2])

    def A3(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF4(V[0]+pure_exc_aff, V[2]+inh_aff, w)-V[3])

    
    def Diff_OP(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A2(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A3(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])

    return Diff_OP
    

def build_up_differential_operator_mossy(TFgrc, TFgoc, TFmli, TFpc, w,
                                         Ngrc, Ngoc, Nmossy, Nmli, Npc, T):
    """
    Implements Equation (3.16) in El BOustani & Destexhe 2009
    in the case of a network of two populations:
    one excitatory and one inhibitory

    Each neuronal population has the same transfer function
    this 2 order formalism computes the impact of finite size effects
    T : is the bin for the Markovian formalism to apply

    the time dependent vector is
     V=[Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm, Vmli, Vpc, Cmlimli, Cmlipc, Cgrcpc, Cgrcmli,
     Cpcpc, Cmligoc, Cmlimossy, Cpcgoc, Cpcmossy]
    the function returns Diff_OP
    and d(V)/dt = Diff_OP(V)
    """

    # dVgrc/dt : grc activity
    def A0(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                .5 * V[7] * diff2_fe_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                .5 * V[5] * diff2_fe_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                .5 * V[5] * diff2_fi_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                .5 * V[6] * diff2_fi_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0])

    # dVgoc/dt : goc activity
    def A1(V, inh_aff=0, pure_exc_aff=0):
        #print('==========================', diff2_fgoc_fgoc_goc(TF2, V[8] + pure_exc_aff, V[0], V[1] + inh_aff))
        return 1. / T * (
                .5 * V[2] * diff2_fgrc_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[7] * diff2_fm_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[4] * diff2_fgrc_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[3] * diff2_fgrc_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[3] * diff2_fgoc_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[5] * diff2_fgoc_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .4 * V[4] * diff2_fm_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[5] * diff2_fm_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                .5 * V[6] * diff2_fgoc_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1])

    # dCgrcgrc/dt : grc variance
    def A2(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                1. / Ngrc * TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) * (
                1. / T - TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w)) +
                (TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) ** 2 +
                2. * V[4] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                2. * V[3] * diff_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                -2. * V[2])

    # dCgrcgoc/dt : grc and goc covariance
    # N.B.: dCgocgrc/dt = dCgrcgoc/dt
    def A3(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) * (
                TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) +
                V[2] * diff_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[3] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                V[3] * diff_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[6] * diff_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                V[5] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[4] * diff_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)+
                - 2. * V[3])

    # dCgrcm/dt : grc and mossy covariance
    def A4(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        return 1 / T *(
                V[5] * diff_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                V[7] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff)
                - 2. * V[4])

    # dCgocm/dt : goc and mossy covariance
    def A5(V, inh_aff=0, pure_exc_aff=0):
        return 1/ T *(
                V[4] * diff_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[5] * diff_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[7] * diff_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                - 2. * V[5])

    # dCgocgoc/dt : goc variance
    def A6(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                1. / Ngoc * TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) * (
                1. / T - TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w)) +
                (TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) ** 2 +
                2. * V[3] * diff_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                2. * V[6] * diff_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                2. * V[5] * diff_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                - 2. * V[6])

    # dCmm/dt : mossy variance
    def A7(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        return 1. / T * (
                    1. / Nmossy * (V[8] * (
                        1. / T - V[8]))
                    -2. * V[7])

    # v mossy
    def A8(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        #return 1. /T * ((V[8]+1 - V[8])/1)
        return V[8]

    # dvmli/dt
    def A9(V, inh_aff=0, pure_exc_aff=0):
        return (1. / T) * (
                .5 * V[2] * diff2_fe_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[14] * diff2_fe_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[14] * diff2_fi_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[11] * diff2_fi_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[9])

    # dvpc/dt
    def A10(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                .5 * V[2] * diff2_fe_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[14] * diff2_fe_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[14] * diff2_fi_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                .5 * V[11] * diff2_fi_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10])

    # dCmlimli/dt
    def A11(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                1. / Nmli * TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) * (
                1. / T - TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w)) +
                (TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[9]) ** 2 +
                2. * V[11] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                2. * V[14] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff)
                -2. * V[11])

    # dCmlipc/ dt
    def A12(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[9]) * (
                TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10]) +
                V[12] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[11] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[13] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[14] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[12])

    # dCgrcpc/ dt
    def A13(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) * (
                TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10]) +
                V[13] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[2] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[18] * diff_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                V[19] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff)
                - 2. * V[13])

    # dCgrcmli/ dt
    def A14(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFgrc(V[8] + pure_exc_aff, V[1] + inh_aff, w) - V[0]) * (
                TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[9]) +
                V[14] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[2] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[16] * diff_fi(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff) +
                V[17] * diff_fe(TFgrc, V[8] + pure_exc_aff, V[1] + inh_aff)
                - 2. * V[14])

    # dCpcpc/dt
    def A15(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                1. / Npc * TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) * (
                1. / T - TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w)) +
                (TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10]) ** 2 +
                2. * V[12] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                2. * V[13] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[15])

    # dCmligoc/dt
    def A16(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFmli(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[9]) * (
                TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) +
                V[14] * diff_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[3] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[16] * diff_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[16] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[17] * diff_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                - 2. * V[16])

    # dCmlimossy/ dt
    def A17(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                V[17] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[4] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[17])


    # dCpcgoc/ dt
    def A18(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                (TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10]) * (
                TFgoc(V[8] + pure_exc_aff, V[0], V[1] + inh_aff, w) - V[1]) +
                V[13] * diff_fgrc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[3] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[18] * diff_fgoc_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff) +
                V[16] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[19] * diff_fm_goc(TFgoc, V[8] + pure_exc_aff, V[0], V[1] + inh_aff)
                - 2. * V[18])

    # dCpcmossy/dt
    def A19(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                V[17] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[4] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[19])


    def Diff_OP(V, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A1(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A2(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A3(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A4(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A5(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A6(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A7(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A8(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A9(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A10(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A11(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A12(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A13(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A14(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A15(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A16(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A17(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A18(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A19(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)
                         ], dtype=object)

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


def find_fixed_point_first_order(TF1, TF2, TF3, TF4,
                                 CI_vec1, t, w, exc_aff, T, num_pop = 4, verbose=True):

    ### FIRST ORDER ##

    X = CI_vec1
    X_vett = np.zeros((len(t), num_pop))
    for i in range(len(t)):
        if i == 0:
            X_vett[0,:]  = X
        else:
            last_X = X
            X = X + (t[1] - t[0]) * build_up_differential_operator_first_order(TF1, TF2, TF3, TF4, w, T)(X, exc_aff[i])

            #print(X)
            #print(last_X)
            X_vett[i, :] = X

    if verbose:
        print('first order prediction: ', X_vett[-1])
    return X_vett, X_vett[-1][0], X_vett[-1][1], X_vett[-1][2], X_vett[-1][3]


def find_fixed_point_mossy(TFgrc, TFgoc, TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Ngoc, Nmossy, Nmli, Npc, T,
                           verbose=False):

    # V = [Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm, Vmli, Vpc, Cmlimli, Cmlipc, Cgrcpc, Cgrcmli,
    #     Cpcpc, Cmligoc, Cmlimossy, Cpcgoc, Cpcmossy]

    ### SECOND ORDER ###

    # simple euler

    X = CI_vec2
    X_vett = np.zeros((len(t), 20))
    for i in range(len(t)):
        last_X = X
        X = X + (t[1] - t[0]) * build_up_differential_operator_mossy(TFgrc, TFgoc, TFmli, TFpc, w,
                                                                Ngrc=Ngrc, Ngoc=Ngoc, Nmossy=Nmossy, Nmli=Nmli, Npc=Npc,
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

def plot_for_thesis_activity(t, X, F_mossy, mytitle, font_size = 8, linew=1.5):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize =(7,5))
    fig.suptitle(mytitle, fontsize = font_size +2)

    # GrC --------------------------------------------------------------------------------------------------------------
    ax1.plot(t, X[:,0], '#62DACF', linewidth = linew)
    ax1.set_ylabel('$\\nu_{GrC}$', fontsize = font_size + 1)
    ax1.set_xticks([])

    # GoC --------------------------------------------------------------------------------------------------------------
    ax2.plot(t, X[:,1], '#7E8ACA', linewidth = linew)
    ax2.set_ylabel('$\\nu_{GoC}$', fontsize = font_size+1)
    ax2.set_xticks([])

    # MLI --------------------------------------------------------------------------------------------------------------
    ax3.plot(t, X[:,9], 'salmon', linewidth = linew)
    ax3.set_ylabel('$\\nu_{MLI}$', fontsize = font_size + 1)
    ax3.set_xticks([])

    # PC --------------------------------------------------------------------------------------------------------------
    ax4.plot(t, X[:,10], 'orchid', linewidth = linew)
    ax4.set_ylabel('$\\nu_{PC}$', fontsize = font_size+1)
    ax4.set_xticks([])

    # Mossy ------------------------------------------------------------------------------------------------------------
    ax5.plot(t, F_mossy, '#D11D12', linewidth = linew)
    ax5.set_ylabel('$\\nu_{drive}$', fontsize = font_size+1)
    ax5.set_xlabel('time [s]', fontsize = font_size+1)

    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.15, top=0.92, bottom = 0.1)
    plt.show()


def plot_for_thesis_GRL_variance(t, X, font_size = 8, linew=1.5):

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

    root_path = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/'
    #root_path = '/home/bcc/bsb-ws/CRBL_MF_Model/'
    NRN1, NRN2, NRN3, NRN4 = 'GrC', 'GoC', 'MLI', 'PC'

    NTWK = 'CRBL_CONFIG_20PARALLEL_wN'#_redGoCGrC'

    """
    FILE_GrC = root_path + '20220525_163030_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    #FILE_GrC = root_path + '20220531_115518_GrC_CRBL_CONFIG_20PARALLEL_wN_redGOCGRC_tsim5_alpha2.0_fit.npy'
    FILE_GoC = root_path + '20220527_231641_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
    #FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/_OLD_TF/20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
    FILE_MLI = root_path + '20220525_163203_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
    FILE_PC = root_path + '20220530_180801_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    """

    FILE_GrC = root_path  + '20220519_120033_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    FILE_GoC= root_path + '20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
    #FILE_MLI = root_path + '20220519_120011_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.6_fit.npy'
    #FILE_PC = root_path + '20220519_120128_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.9_fit.npy' #STANDARD CON PAUSA BELLA

    #FILE_PC = root_path + '20220608_121728_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha4.0_fit.npy'
    #FILE_PC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220608_121615_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha4.0_fit.npy'
    #FILE_PC =  root_path + '20220608_124413_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha3.0_fit.npy'
    #FILE_PC = root_path + 'numerical_network/20220619_155605_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'

    #FILE_PC = root_path + 'numerical_network/20220619_180012_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    #FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/numerical_network/20220619_155605_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'

    #Trial Ie*1.2 and alpha PREDICTION 5 for MLI and PC
    FILE_MLI = root_path + '20220622_085550_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.8_fit.npy'
    #FILE_MLI = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220628_194720_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.2_fit.npy'
    #FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220622_085610_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    FILE_PC = root_path + '20220622_085610_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.5_fit.npy'

    #FILE_MLI = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220622_171304_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.8_fit.npy'
    #FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220622_171355_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha3.0_fit.npy'

    TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=2.0)
    TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.3)
    TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=5)
    TFpc = load_transfer_functions(NRN4, NTWK, FILE_PC, alpha=5) # !!!!!Cambiato per avere pc con giusto FR 4


    w = 0
    #t = np.arange(5000) * 1e-4
    dt = 1e-4
    #t = np.arange(0, 5+dt, dt)
    T = 3.5e-3

    #t = np.arange(0, 1.5 + dt, dt)


    t = np.arange(0, 0.5, dt)
    f_backnoise = np.random.rand(len(t))*4 #* 4  # + np.ones(len(t))*50

    f_tone1 = rect_input(time=t, t_start=1250, t_end=3750, minval=0, freq=50, noise_freq=0)

    #f_tone1 = rect_input(time=t, t_start=3500, t_end=6000, minval=0, freq=50, noise_freq=0)
    #f_tone2 = rect_input(time=t, t_start=6500, t_end=9000, minval=0, freq=50, noise_freq=0)
    #f_tone3 = rect_input(time=t, t_start=9500, t_end=12000, minval=0, freq=50, noise_freq=0)
    #fmossy = f_backnoise + f_tone1 + f_tone2 + f_tone3

    t = np.arange(0, 0.5, dt)
    f_backnoise = np.random.rand(len(t))*4 #* 4  # + np.ones(len(t))*50

    """
    f_tone1 = rect_input(time=t, t_start=3500, t_end=6000, minval=0, freq=50, noise_freq=0)
    f_tone2 = rect_input(time=t, t_start=6500, t_end=9000, minval=0, freq=50, noise_freq=0)
    f_tone3 = rect_input(time=t, t_start=9500, t_end=12000, minval=0, freq=50, noise_freq=0)
    fmossy = f_backnoise + f_tone1 + f_tone2 + f_tone3
>>>>>>> 6ac0cc9b1397f83bc78089029f90126a907d1bef

    # pulse period = quanti ne voglio, pulse_width = quanta larghi
    #QUANTI =  1600/500 = Length/pulse_period.
    #AMPIEZZA = pulse_width - se = 100 --> 10 ms (pulse_width*dt)
    #fmossy= impulse_train(time=t, pulse_period=500, pulse_width=200, pulse_max_freq=50, pulse_baseline=0, noise_freq=4) #10 up 20 down

    #fmossy = impulse_train_s500(time=t, pulse_period=500, pulse_width=200, pulse_max_freq=50, pulse_baseline=0,
    #                      noise_freq=4)  # 10 up 20 down


    # json file: background noise
    #Whisker stim
    freq, amplitude = 6, 20 #to have 40 ampl
    f_sin = amplitude * np.sin(2 * np.pi * freq * t) + amplitude #add amplitude value to avoid negative values
    #fmossy = f_sin + f_backnoise
    fmossy = f_backnoise + f_sin #+ f_tone2 + f_tone3

    """
    amplitude = 7.5 # 20
    freq1 = 15
    freq2 = 30
    freq3 = 1
    f_sin1 = amplitude * np.sin(2 * np.pi * freq1 * t) + amplitude/2  # add amplitude value to avoid negative values
    f_sin2 = amplitude * np.sin(2 * np.pi * freq2 * t) + amplitude
    f_sin3 = amplitude * np.sin(2 * np.pi * freq3 * t) + amplitude
    fmossy = f_sin1 + f_sin2 + f_sin3 + f_backnoise  # + f_sin_bis

    #fmossy = fmossy2+fmossy1
    """

    #CI_vec1 = [0.5, 10, 8.5, 10]

    ## First order check
    #X1, fix_grc, fix_goc, fix_mli, fix_pc = find_fixed_point_first_order(TFgrc, TFgoc, TFmli, TFpc, CI_vec1, t, w, fmossy, T)
    #plot_MF(t,  np.array([X1[:,0], X1[:,1], X1[:,2], X1[:,3], fmossy[:]]), 'First Order MF', '')
    """

    Ngrc = 28615
    Ngoc = 70
    Nmossy = 2336
    Nmli = 299+147
    Npc = 99

    #fmossy = np.ones(len(t)) * 50 + np.random.rand(len(t)) * 4
    #fmossy[:30] = np.random.rand(30) * 4

    tstart_sim = time.time()
    #CI_vec2 = [0.5, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0], 8.5, 20, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    CI_vec2 = [0.5, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0], 15, 38, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # we start from the first order prediction !!!
    #CI_vec2 = [fix_grc, fix_goc, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0], fix_mli, fix_pc, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    X = find_fixed_point_mossy(TFgrc, TFgoc, TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Ngoc,
                               Nmossy, Nmli, Npc, T, verbose=False)

    time_sim = tstart_sim - time.time()
    print('simulation last: ',time_sim)


    title_sim = 'prova'
    #title_sim = '4paper_updown_500ms.npy'
    #title_sim='4paper_updown3_1500'
    #title_sim = '4paper_updown_500ms.npy'



    plot_MF(t, np.array([X[:, 0], X[:, 1], X[:, 9], X[:, 10],  X[:,8]]), title_sim, title_sim)

    #plot_MF(t, np.array([X[3000:, 0], X[3000:, 1], X[3000:, 9], X[3000:, 10], fmossy[3000:]]),
    #                 'Second Order MF', 'MF2ORD_ALLPARALL_KNsyn')

    #SALVO COSI' PER PLOT!!!!!!!!!! mossy alla fineeeeeee
    np.save(title_sim+'.npy', [X[:,0], X[:,1], X[:,9], X[:,10], X[:,8]])

    #plot_for_thesis_1vs2order(t, X1, X, fmossy)
    #mytitle = 'CRBL MF - 2nd Order'
    #plot_for_thesis_activity(t, X, fmossy, mytitle)


    #plot_for_thesis_GRL_variance(t, X)
