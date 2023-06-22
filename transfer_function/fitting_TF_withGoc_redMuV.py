""" Fitting of TF Semi-Analytic Expression"""
import numpy as np
import scipy.special as sp_spec
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from CRBL_MF_Model.graph_utils.my_graph import build_bar_legend, set_plot



## =======================
## ===== Parameters =====
## =======================

def pseq_params_eglif_goc(params):

    Qe_g, Qe_m = params['Qe_g'], params['Qe_m']
    Te_g, Te_m, Ee = params['Te_g'], params['Te_m'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']

    Ke_g = params['Ke_g']
    Ke_m = params['Ke_m']
    Ki = params['Ki']

    return Qe_g, Qe_m, Te_g, Te_m, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ke_m, Ki


## ===========================================
## ===== Membrane fluctuation properties =====
## ===========================================
# based on shotnoise theory, computation of the three statistical moment_
# muV, sV, Tv (Ref.: Zerlaut et al. 2018)

def get_fluct_regime_varsup_eglif_goc(Fe_m, Fe_g, Fi, XX,
                                  Qe_g, Qe_m, Te_g, Te_m, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ke_m, Ki, P0):
    fe_m = Fe_m
    fe_g = Fe_g
    fi = Fi

    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe_g, muGe_m, muGi = Qe_g * Ke_g * Te_g * fe_g, Qe_m * Ke_m * Te_m * fe_m, Qi * Ki * Ti * fi #EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe_g + muGe_m + muGi #EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (np.e * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - XX) / muG  # XX = adaptation


    # ---------------------------- Reduction of space parameters -------------------------------------
    """muV = np.concatenate([muV_glob[:7, :4, 0:4], muV_glob[:7, 4:8, 4:8], muV_glob[7:, 8:12, 8:12],
                          muV_glob[7:, 12:16, 12:16], muV_glob[7:, 16:20, 16:20]], axis=-1)

    muG = np.concatenate([muG[:7, :4, 0:4], muG[:7, 4:8, 4:8], muG[7:, 8:12, 8:12],
                          muG[7:, 12:16, 12:16], muG[7:, 16:20, 16:20]], axis=-1)"""

    """
    print('All params Baaaaaby:')
    print('Qe_g: ',Qe_g,'\tTe_g: ',Te_g,'\tEe: ',Ee,'\tKe_g: ',Ke_g,
          '\nQe_m: ', Qe_m, '\tTe_m: ', Te_m,'\tKe_m: ', Ke_m,
          '\nQi: ',Qi,'\tTi: ',Ti,'\tEi: ',Ei,'\tKi: ',Ki,
          '\nGl: ',Gl,'\t Cm: ',Cm,'\tEl: ',El)

    print('Conductances')
    print('muGe_g: ', muGe_g,'\tmuGe_m: ',muGe_m,'\tmuGi: ',muGi)
    print('MuV without adap: ', np.e * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) / muG  )
    print('MuV with adap: ', muV)
    """

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue_g, Ue_m, Ui = Qe_g / muG * (Ee - muV), Qe_m / muG * (Ee - muV), Qi / muG * (Ei - muV) #EQUAL TO EXP

    #sVe= (2*Tm + Te) * (np.e * Ue *Te) ** 2 / (2* (Te + Tm)) **2 *Ke * fe
    #sVi = (2*Tm + Ti) * (np.e * Ui *Ti) ** 2 / (2* (Ti + Tm)) **2 *Ki * fi

    sVe_g = (2 * Tm + Te_g) * ((np.e * Ue_g * Te_g)/ (2 * (Te_g + Tm))) ** 2 * Ke_g * fe_g
    sVe_m = (2 * Tm + Te_m) * ((np.e * Ue_m * Te_m) / (2 * (Te_m + Tm))) ** 2 * Ke_m * fe_m
    sVi = (2 * Tm + Ti) * ((np.e * Ui *Ti) / (2* (Ti + Tm))) **2 *Ki * fi

    sV = np.sqrt(sVe_g + sVe_m + sVi)

    fe_m, fe_g, fi = fe_m + 1e-15, fe_g + 1e-15, fi + 1e-15  # just to insure a non zero division

    Tv_num= Ke_g * fe_g * Ue_g ** 2 * Te_g ** 2 * np.e ** 2 + \
            Ke_m * fe_m * Ue_m ** 2 * Te_m ** 2 * np.e ** 2 + \
            Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sV+1e-20) ** 2)


    TvN = Tv * Gl / Cm  # normalization

    return muGe_g, muGe_m, muGi, muG, muV, sV+1e-20, muGn, TvN, Tv


## ==================================
## === Phenomenological Threshold ===
## ==================================


# EQUATION OF ERFC = vout = Fout = TF expression (Zerlaut et al. 2018)
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm, alpha):
    return .5 / TvN * Gl / Cm * (sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sV)) * alpha


# EQUATION OF VOUT obtained by inverting erfc_func above (Zerlaut et al. 2016)
# Used in first step of fitting procedure (fitting of V eff thre expression)
# Y = TF numerical template
def effective_Vthre(Y, muV, sV, TvN, Gl, Cm, alpha):
    Vthre_eff = muV + np.sqrt(2) * sV * sp_spec.erfcinv((1 / alpha) * (Y * 2 * TvN * Cm / Gl))  # effective threshold

    return Vthre_eff


# Polynomia expression of V eff thre - initial condition
def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4):
    """
    setting by default to True the square
    because when use by external modules, coeff[5:]=np.zeros(3)
    in the case of a linear threshold
    """

    muV0, DmuV0 = -60e-3, 10e-3
    sV0, DsV0 = 4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.

    return P0 + P1 * (muV - muV0) / DmuV0 + \
           P2 * (sV - sV0) / DsV0 + P3 * (TvN - TvN0) / DTvN0 + P4 * np.log(muGn)
    # return P0+0*muV


#def alpha_gain():



## =================================
## ==== TF Polynomial Expression ===
## =================================
# Definition of the TF TEMPLATE
# It's called IN FITTING routine - MINIMIZATION

def TF_my_templateup_eglif_goc(fe_m, fe_g, fi, XX, alpha, Qe_g, Qe_m, Te_g, Te_m, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ke_m, Ki,
                            P0, P1, P2, P3, P4):
    # here TOTAL (sum over synapses) excitatory and inhibitory input

    if (hasattr(fe_m, "__len__")):
        fe_m[fe_m < 1e-8] = 1e-8
    else:
        if (fe_m < 1e-8):
            fe_m = 1e-8
    if (hasattr(fe_g, "__len__")):
        fe_g[fe_g < 1e-8] = 1e-8
    else:
        if (fe_g < 1e-8):
            fe_g = 1e-8
    if (hasattr(fi, "__len__")):
        fi[fi < 1e-8] = 1e-8
    else:
        if (fi < 1e-8):
            fi = 1e-8

    muGe_g, muGe_m, muGi, muG, muV, sV, muGn, TvN, Tv = get_fluct_regime_varsup_eglif_goc(fe_m, fe_g, fi, XX, Qe_g, Qe_m,
                                                                                          Te_g, Te_m, Ee, Qi, Ti,
                                                                                          Ei, Gl, Cm, El, Ke_g,
                                                                                          Ke_m, Ki, P0)  # , P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)


    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4)


    if (hasattr(muV, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sV[sV < 1e-4] = 1e-4
    else:
        if (sV < 1e-4):
            sV = 1e-4

    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm, alpha)

    if (hasattr(Fout_th, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th < 1e-8] = 1e-8
    else:
        if (Fout_th < 1e-8):
            Fout_th = 1e-8

    return Fout_th


## ===============================
## ====== 2 Step Fitting  ========
## ===============================
# 1) Vthre
# 2) vout (using Vthre 1) just optimized)

def fitting_Vthre_then_Fout_eglif_goc(Fout, Fe_m_eff, Fe_g_eff, fiSim, w, params, alpha,
                                  maxiter=50000, xtol=1e-5, with_square_terms=True):
    Gl, Cm, El = params['Gl'], params['Cm'], params['El']

    muGe_g, muGe_m, muGi, muG, muV, sV, muGn, TvN, Tv = get_fluct_regime_varsup_eglif_goc(Fe_m_eff, Fe_g_eff, fiSim, w,
                                                                            *pseq_params_eglif_goc(params), -50e-3)

    foutlim = 30
    #foutlim= Fout.max()

    i_non_zeros = np.where((Fout > 0.) & (Fout < foutlim))  # GrC - based on trial and error
    print("Fout Limit: ", foutlim)

    Vthre_eff = effective_Vthre(Fout[i_non_zeros], muV[i_non_zeros],
                                sV[i_non_zeros], TvN[i_non_zeros], params['Gl'], params['Cm'], alpha[i_non_zeros])

    P = [-50e-3, 0, 0, 0, 0]
    # P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    print("OPTIMIZATION STEP 1 ==========================================\nVthre Optimization")

    def Res(p):
        vthre = threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN[i_non_zeros],
                               muGn[i_non_zeros], *p)
        return np.mean((Vthre_eff - vthre) ** 2)

    # plsq = minimize(Res, P, method='SLSQP', options={'ftol': 1e-15, 'disp': True, 'maxiter': 40000})

    plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x

    print('========== P output from Vthreshold opt: ', P)

    print("OPTIMIZATION STEP 2 ==========================================\nFout Optimization")

    def Res(p):
        return np.mean((Fout -
                        TF_my_templateup_eglif_goc(Fe_m_eff, Fe_g_eff, fiSim, w, alpha,
                                               params['Qe_g'], params['Qe_m'], params['Te_g'], params['Te_m'],
                                               params['Ee'], params['Qi'], params['Ti'], params['Ei'], params['Gl'],
                                               params['Cm'], params['El'], params['Ke_g'], params['Ke_m'],
                                               params['Ki'], *p)) ** 2)


    plsq = minimize(Res, P, method='nelder-mead',
                    options={'xtol': xtol, 'disp': True, 'maxiter': maxiter})

    # plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x
    print('========== P output from Fout opt: ', P)

    print("END OF OPTIMIZATION PROCESS ==========================================")

    params['P'] = P

    # thrplot = threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN[i_non_zeros], muGn[i_non_zeros], *P)
    # plt.figure()
    # for i in range(Fe_eff.shape[0]):
    #    plt.plot(Fe_eff[i,:], erfc_func(muV[i,:], sV[i,:], TvN[i,:], thrplot[i,:], Gl, Cm), color=plt.cm.viridis(i/Fe_eff.shape[0]))
    # plt.show()

    # plt.plot(muV, erfc_func(muV, 4e-3, 0.5, thrplot, Gl, Cm), 'bd')
    # plt.show()

    return P

## ===============================
## ========= Actual Fit  =========
## ===============================
# Load num template computed with EGLIF_num_TF.py
# and implement the fitting procedure


def load_my_data_goc(FOLDER, adap_bool):

    MEANfreq = np.load(FOLDER + '/numTF.npy', allow_pickle=True)
    sd_freq = np.load(FOLDER + '/FoutSD.npy', allow_pickle=True)
    Fi = np.load(FOLDER + '/fi_vector.npy', allow_pickle=True)
    Fe_m = np.load(FOLDER + '/fe_m_vector.npy', allow_pickle=True)
    Fe_g = np.load(FOLDER + '/fe_g_vector.npy', allow_pickle=True)
    params = np.load(FOLDER + '/params.npy', allow_pickle=True).item()
    sim_params = np.load(FOLDER + '/sim_len.npy', allow_pickle=True)

    if adap_bool:
        w = np.load(FOLDER + '/adaptation.npy', allow_pickle=True)
    else:
        w = np.zeros((len(Fi), len(Fe_g), len(Fe_m)))


    print('Adap: ',w)
    fiSim = np.zeros((len(Fi), len(Fe_g), len(Fe_m)))
    Fe_m_eff = np.zeros((len(Fi), len(Fe_g), len(Fe_m)))
    Fe_g_eff = np.zeros((len(Fi), len(Fe_g), len(Fe_m)))

    alpha = np.ones((len(Fi), len(Fe_g), len(Fe_m))) #alpha init = 1

    for fe_m in range(len(Fe_m)):
        for fe_g in range(len(Fe_g)):
            fiSim[:, int(fe_g), int(fe_m)] = Fi[:]

    for fe_g in range(len(Fe_g)):
        for fi in range(len(Fi)):
            Fe_m_eff[int(fi), int(fe_g), :] = Fe_m[:]

    for fe_m in range(len(Fe_m)):
        for fi in range(len(Fi)):
            Fe_g_eff[int(fi), :, int(fe_m)] = Fe_g[:]

    #for fe_g in range(len(Fe_g)):
    #    if Fe_g[int(fe_g)] >= 40:
    #        alpha[:, fe_g, :] = alpha[:, fe_g, :] *2

    for fe_g in range(len(Fe_g)):
        #sigmoid
        alpha[:, fe_g, :] = 1/(1+np.exp(-Fe_g[fe_g]))*4

    return MEANfreq, sd_freq, w, fiSim, Fe_m_eff, Fe_g_eff, params, sim_params, alpha


def reduce_params_space(FOLDER, adap_bool):

    MEANfreq, sd_freq, w, fi, fe_m, fe_g, params, sim_params, alpha = load_my_data_goc(FOLDER, adap_bool)

    #Reduction of Parameters space:
    fe_g = np.concatenate([fe_g[:7, :4, 0:4], fe_g[:7, 4:8, 4:8], fe_g[7:, 8:12, 8:12],
                           fe_g[7:, 12:16, 12:16], fe_g[7:, 16:20, 16:20]], axis=-1)

    fe_m= np.concatenate([fe_m[:7, :4, 0:4], fe_m[:7, 4:8, 4:8], fe_m[7:, 8:12, 8:12],
                          fe_m[7:, 12:16, 12:16], fe_m[7:, 16:20, 16:20]], axis=-1)

    fi = np.concatenate([fi[:7, :4, 0:4], fi[:7, 4:8, 4:8], fi[7:, 8:12, 8:12],
                         fi[7:, 12:16, 12:16], fi[7:, 16:20, 16:20]], axis=-1)

    MEANfreq = np.concatenate([MEANfreq[:7, :4, 0:4],  MEANfreq[:7, 4:8, 4:8],  MEANfreq[7:, 8:12, 8:12],
                            MEANfreq[7:, 12:16, 12:16], MEANfreq[7:, 16:20, 16:20]], axis=-1)

    w = np.concatenate([w[:7, :4, 0:4],  w[:7, 4:8, 4:8],  w[7:, 8:12, 8:12],
                            w[7:, 12:16, 12:16], w[7:, 16:20, 16:20]], axis=-1)

    sd_freq = np.concatenate([sd_freq[:7, :4, 0:4],  sd_freq[:7, 4:8, 4:8],  sd_freq[7:, 8:12, 8:12],
                                sd_freq[7:, 12:16, 12:16], sd_freq[7:, 16:20, 16:20]], axis=-1)

    alpha = np.concatenate([alpha[:7, :4, 0:4],  alpha[:7, 4:8, 4:8],  alpha[7:, 8:12, 8:12],
                                alpha[7:, 12:16, 12:16], alpha[7:, 16:20, 16:20]], axis=-1)

    return MEANfreq, sd_freq, w, fi, fe_m, fe_g, params, sim_params, alpha



def make_fit_from_data_eglif_goc(FOLDER, adap_bool):

    MEANfreq, sd_freq, w, fi, fe_m, fe_g, params, sim_params, alpha = load_my_data_goc(FOLDER, adap_bool)

    #MEANfreq, sd_freq, w, fi, fe_m, fe_g, params, sim_params, alpha = reduce_params_space(FOLDER, adap_bool)


    P = fitting_Vthre_then_Fout_eglif_goc(MEANfreq, fe_m, fe_g, fi, w, params, alpha)

    print('==================================================')
    print(1e3 * np.array(P), 'Fitted P in mV')

    # then we save it:
    filename = FOLDER + '_fit.npy'
    print('coefficients P saved in ', filename)
    np.save(filename, np.array(P))

    return P