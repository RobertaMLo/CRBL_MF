import numpy as np
import sys
sys.path.append('../')
from MF_prediction import *
from MF_prediction.load_config_TF import *
from MF_prediction.input_library import *

import matplotlib

import matplotlib.pyplot as plt
import os

import time

def build_up_differential_operator_first_order(TF3, TF4, w, T):
    """
    simple first order system
    """

    def A2(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF3(V[0]+pure_exc_aff, V[2]+inh_aff,  w)-V[2])

    def A3(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return (1./T)*(TF4(V[0]+pure_exc_aff, V[2]+inh_aff, w)-V[3])

    
    def Diff_OP(V, exc_aff, inh_aff=0, pure_exc_aff=0):
        return np.array([
                         A2(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A3(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])

    return Diff_OP
    

def build_up_differential_operator_mossy(TFmli, TFpc, w,
                                         Ngrc, Nmli, Npc, T):
    """
    Implements Equation (3.16) in El BOustani & Destexhe 2009
    in the case of a network of two populations:
    one excitatory and one inhibitory

    Each neuronal population has the same transfer function
    this 2 order formalism computes the impact of finite size effects
    T : is the bin for the Markovian formalism to apply

    the time dependent vector vector is
     V=[Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm, Vmli, Vpc, Cmlimli, Cmlipc, Cgrcpc, Cgrcmli,
     Cpcpc, Cmligoc, Cmlimossy, Cpcgoc, Cpcmossy]
    the function returns Diff_OP
    and d(V)/dt = Diff_OP(V)
    """


    # dCmm/dt : mossy variance
    def A7(V, inh_aff=0, pure_exc_aff=0):
        #print(V[8])
        return 1. / T * (
                    1. / Ngrc * (V[8] * (
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

    # dCpcpc/dt
    def A15(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                1. / Npc * TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) * (
                1. / T - TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w)) +
                (TFpc(V[0] + pure_exc_aff, V[9] + inh_aff, w) - V[10]) ** 2 +
                2. * V[12] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                2. * V[13] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[15])

    # dCmlimossy/ dt
    def A17(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                V[17] * diff_fi(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[4] * diff_fe(TFmli, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[17])

    # dCpcmossy/dt
    def A19(V, inh_aff=0, pure_exc_aff=0):
        return 1. / T * (
                V[17] * diff_fi(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff) +
                V[4] * diff_fe(TFpc, V[0] + pure_exc_aff, V[9] + inh_aff)
                - 2. * V[19])


    def Diff_OP(V, inh_aff=0, pure_exc_aff=0):
        return np.array([
                         A7(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A8(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A9(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A10(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A11(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A12(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A15(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
                         A17(V, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),
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


def find_fixed_point_first_order(TF3, TF4,
                                 CI_vec1, t, w, exc_aff, T, num_pop = 4, verbose=True):

    ### FIRST ORDER ##

    X = CI_vec1
    X_vett = np.zeros((len(t), num_pop))
    for i in range(len(t)):
        if i == 0:
            X_vett[0,:]  = X
        else:
            last_X = X
            X = X + (t[1] - t[0]) * build_up_differential_operator_first_order(TF3, TF4, w, T)(X, exc_aff[i])

            X_vett[i, :] = X

    return X_vett


def MF_second_order(TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Nmli, Npc, T,
                           verbose=False):

    # V = [Cmm, Vm, Vmli, Vpc, Cmlimli, Cmlipc, Cpcpc, Cmlimossy, Cpcmossy]

    ### SECOND ORDER ###

    # simple euler

    X = CI_vec2
    X_vett = np.zeros((len(t), 20))
    for i in range(len(t)):
        last_X = X
        X = X + (t[1] - t[0]) * build_up_differential_operator_mossy(TFmli, TFpc, w,
                                                                Ngrc=Ngrc, Nmli=Nmli, Npc=Npc,
                                                                T=T)(X)
        #X[8] = fmossy[i]/T
        X[8] = fmossy[i]
        #X[4], X[5], X[7] = 0, 0, 0
        X_vett[i, :] = X

    return X_vett
    
def plot_MF(t, GrC, MLI, PC, mytitle):
    t = t*1000
    t_max = np.max(t)
    fig = make_subplots(rows=3, cols=1,
                        x_title='time [ms]',
                        y_title='Population activity [Hz]',
                        subplot_titles=('PC', 'MLI', 'GrC (driving)'))

    fig.add_trace(go.Scatter(x=t, y=GrC, mode='lines', name='$\\nu_{GrC}$',  line=dict(color = "red")),
                      row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=MLI, mode='lines', name='$\\nu_{MLI}$', line=dict(color = "salmon")),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=PC, mode='lines', name='$\\nu_{PC}$', line=dict(color = "green")),
                      row=1, col=1)



    fig.update_layout(title=mytitle)

    fig.update_layout(
        title=mytitle,
        autosize=True,
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[x_i for x_i in xval],
        #ticktext=[str(int(x_i * 100)) + '%' for x_i in xval],
        # tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle=0,
        title_standoff=25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        #ticktext=[str(round(y_i, 2)) for y_i in yval],
        tickangle=0,
        #title_text=ytitle,
        #title_font={"size": fontsize},
        title_standoff=25)

    fig.show()


