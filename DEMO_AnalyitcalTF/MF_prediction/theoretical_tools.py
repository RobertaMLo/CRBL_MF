import numpy as np
import scipy.special as sp_spec

# Parameters import: DIFFERENT FROM THE OTHER --> IMPORT ALSO FITTED P!
def pseq_params(params):

    if 'P' in params.keys():
        P = params['P']
    else: # no correction
        P = [-45e-3]
        for i in range(1,11):
            P.append(0)

    return params['Qe'], params['Te'], params['Ee'], params['Qi'], params['Ti'], params['Ei'], params['Gl'], \
           params['Cm'], params['El'], params['Ke'], params['Ki'], P[0], P[1], P[2], P[3], P[4]


# Computation of fluc properties: Change just the RETURN
def get_fluct_regime_varsup_eglif(Fe, Fi, XX,
                                  Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki, P0, P1, P2, P3, P4):
    fe = Fe
    fi = Fi

    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe, muGi = Qe * Ke * Te * fe, Qi * Ki * Ti * fi #EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe + muGi #EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (np.e * (muGe * Ee + muGi * Ei + Gl * El) - XX) / muG  # XX = adaptation

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue, Ui = Qe / muG * (Ee - muV), Qi / muG * (Ei - muV) #EQUAL TO EXP

    #sVe= (2*Tm + Te) * (np.e * Ue *Te) ** 2 / (2* (Te + Tm)) **2 *Ke * fe
    #sVi = (2*Tm + Ti) * (np.e * Ui *Ti) ** 2 / (2* (Ti + Tm)) **2 *Ki * fi

    sVe = (2 * Tm + Te) * ((np.e * Ue * Te) / (2 * (Te + Tm))) ** 2 * Ke * fe
    sVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2 * (Ti + Tm))) ** 2 * Ki * fi

    sV = np.sqrt(np.abs(sVe + sVi))
    #sV = np.sqrt(sVe + sVi)

    fe, fi = fe + 1e-9, fi + 1e-9  # just to insure a non zero division

    Tv_num= Ke * fe * Ue ** 2 * Te ** 2 * np.e ** 2 + Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sV+1e-20) ** 2)


    TvN = Tv * Gl / Cm  # normalization

    return muV, sV+1e-20, muGn, TvN


# Mean and Variances of the conductance
def mean_and_var_conductance(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki, P0, P1, P2, P3, P4):
    muGe, muGi = Qe * Ke * Te * fe, Qi * Ki * Ti * fi
    svGe, svGi = Qe * np.sqrt(Te * fe * Ke/2.), Qi * np.sqrt(Ti * fi * Ki/2.),

    return muGe, muGi, svGe,


#TF semi-analytic expression
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm, alpha):
    return .5/TvN * Gl/Cm * (sp_spec.erfc( (Vthre-muV)/np.sqrt(2)/sV)) * alpha

#Veff_thre as inverse function of TF semi-analytic expression
def effective_Vthre(Y, muV, sV, TvN, Gl, Cm, alpha):
    Vthre_eff = muV + np.sqrt(2) * sV * sp_spec.erfcinv( (1/alpha) * (Y*2*TvN*Cm/Gl) ) # effective threshold
    return Vthre_eff

#Veff_thre as polynomial expression
def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4):
    muV0, DmuV0 = -60e-3, 10e-3
    sV0, DsV0 = 4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    return P0 + P1 * (muV - muV0) / DmuV0 + \
           P2 * (sV - sV0) / DsV0 + P3 * (TvN - TvN0) / DTvN0 + P4 * np.log(muGn)

# Final transfer function template :
def TF_my_template(fe, fi, XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki, P0, P1, P2, P3, P4, alpha):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    muV, sV, muGn, TvN = get_fluct_regime_varsup_eglif(fe, fi, XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki,
                                                       P0, P1, P2, P3, P4)

    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4)
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm, alpha)
    return Fout_th
    
