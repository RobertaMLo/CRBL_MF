""" Fitting of TF Semi-Analytic Expression"""
import numpy as np
import scipy.special as sp_spec
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from graph_utils.my_graph import build_bar_legend, set_plot, get_linear_colormap


## =======================
## ===== Parameters =====
## =======================
def pseq_params_eglif(params):
    Qe = params['Qe']
    Te, Ee = params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm, El = params['Gl'], params['Cm'], params['El']

    Ke = params['Ke']
    Ki = params['Ki']

    return Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki


## ===========================================
## ===== Membrane fluctuation properties =====
## ===========================================
# based on shotnoise theory, computation of the three statistical moment_
# muV, sV, Tv (Ref.: Zerlaut et al. 2018)
def get_fluct_regime_varsup_eglif(Fe, Fi, XX,
                                  Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki, P0):
    fe = Fe
    fi = Fi

    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe, muGi = Qe * Ke * Te * fe, Qi * Ki * Ti * fi  # EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe + muGi  # EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    """
    print('All params Baaaaaby:')
    print('Qe: ',Qe,'\tTe: ',Te,'\tEe: ',Ee,'\tKe: ',Ke,
          '\nQi: ',Qi,'\tTi: ',Ti,'\tEi: ',Ei,'\tKi: ',Ki,
          '\nGl: ',Gl,'\t Cm: ',Cm,'\tEl: ',El)

    print('Conductances')
    print('muGe: ', muGe,'\tmuGi: ',muGi)
    """
    muV = (np.e * (muGe * Ee + muGi * Ei + Gl * El) - XX) / muG  # XX = adaptation

    """
    print('MuV without adap: ', (np.e * (muGe * Ee + muGi * Ei + Gl * El) / muG) )
    print('MuV with adap: ', muV)
    """

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue, Ui = Qe / muG * (Ee - muV), Qi / muG * (Ei - muV)  # EQUAL TO EXP

    # sVe= (2*Tm + Te) * (np.e * Ue *Te) ** 2 / (2* (Te + Tm)) **2 *Ke * fe
    # sVi = (2*Tm + Ti) * (np.e * Ui *Ti) ** 2 / (2* (Ti + Tm)) **2 *Ki * fi

    sVe = (2 * Tm + Te) * ((np.e * Ue * Te) / (2 * (Te + Tm))) ** 2 * Ke * fe
    sVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2 * (Ti + Tm))) ** 2 * Ki * fi

    sV = np.sqrt(sVe + sVi)

    fe, fi = fe + 1e-9, fi + 1e-9  # just to insure a non zero division

    Tv_num = Ke * fe * Ue ** 2 * Te ** 2 * np.e ** 2 + Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sV + 1e-20) ** 2)

    TvN = Tv * Gl / Cm  # normalization

    #plot_MF_statistics(muGe, muGi, muG, muV, sV, Tv)

    return muGe, muGi, muG, muV, sV + 1e-20, muGn, TvN


def plot_MF_statistics(muGe, muGi, muG, muV, sV, Tv):
    media = sns.heatmap(muV)
    plt.title('Average Membrane Potential [V]')
    plt.xlabel('Fe [Hz]')
    plt.ylabel('Fi [Hz]')
    plt.show()
    var = sns.heatmap(sV)
    plt.title('Membrane Potential StDev [V]')
    plt.xlabel('Fe [Hz]')
    plt.ylabel('Fi [Hz]')
    plt.show()
    tau = sns.heatmap(Tv)
    plt.title('Autocorrelation Time [s]')
    plt.xlabel('Fe [Hz]')
    plt.ylabel('Fi [Hz]')
    plt.show()
    mug = sns.heatmap(muG)
    plt.title('Target neuron conductance [S]')
    plt.xlabel('Fe [Hz]')
    plt.ylabel('Fi [Hz]')
    plt.show()


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


## =================================
## ==== TF Polynomial Expression ===
## =================================
# Definition of the TF TEMPLATE
# It's called IN FITTING routine - MINIMIZATION
def TF_my_templateup_eglif(fe, fi, XX, alpha, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki,
                           P0, P1, P2, P3, P4):
    # here TOTAL (sum over synapses) excitatory and inhibitory input

    if (hasattr(fe, "__len__")):
        fe[fe < 1e-8] = 1e-8
    else:
        if (fe < 1e-8):
            fe = 1e-8
    if (hasattr(fi, "__len__")):
        fi[fi < 1e-8] = 1e-8
    else:
        if (fi < 1e-8):
            fi = 1e-8

    muGe, muGi, muG, muV, sV, muGn, TvN = get_fluct_regime_varsup_eglif(fe, fi, XX, Qe, Te, Ee,
                                                                            Qi, Ti, Ei, Gl, Cm, El,
                                                                            Ke, Ki,
                                                                            P0)

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
## ====== 2-Step Fitting  ========
## ===============================
# 1) Vthre
# 2) vout (using Vthre 1) just optimized)
def fitting_Vthre_then_Fout_eglif(muGe, muGi, muG, muV, sV, muGn, TvN, Fout, Fe_eff, fiSim, w, params, alpha,
                                  maxiter=50000, xtol=1e-5):
    Gl, Cm, El = params['Gl'], params['Cm'], params['El']

    foutlim= Fout.max()

    i_non_zeros = np.where((Fout > 0.) & (Fout < foutlim))
    print("Fout Limit: ", foutlim)

    Vthre_eff = effective_Vthre(Fout[i_non_zeros], muV[i_non_zeros], \
                                sV[i_non_zeros], TvN[i_non_zeros], params['Gl'], params['Cm'], alpha)

    P = [-50e-3, 0, 0, 0, 0]

    print("OPTIMIZATION STEP 1 ==========================================\nVthre Optimization")

    def Res(p):
        vthre = threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN[i_non_zeros],
                               muGn[i_non_zeros], *p)
        return np.mean((Vthre_eff - vthre) ** 2)


    plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x

    print('========== P output from Vthreshold opt: ', P)

    print("OPTIMIZATION STEP 2 ==========================================\nFout Optimization")

    def Res(p):
        return np.mean((Fout -
                        TF_my_templateup_eglif(Fe_eff, fiSim, w, alpha,
                                               params['Qe'], params['Te'], params['Ee'], params['Qi'], params['Ti'],
                                               params['Ei'], params['Gl'], params['Cm'], params['El'], params['Ke'],
                                               params['Ki'], *p)) ** 2)

    plsq = minimize(Res, P, method='nelder-mead',
                    options={'xtol': xtol, 'disp': True, 'maxiter': maxiter})

    P = plsq.x
    print('========== P output from Fout opt: ', P)

    print("END OF OPTIMIZATION PROCESS ==========================================")

    params['P'] = P

    return P

## ===============================
## ========= Actual Fit  =========
## ===============================
# Load Numerical templates


def plot_TF_numerical_vs_analytical_2D(numTF, SDfreq, fiSim, Fe_eff_grc, w,
                                                     P, alpha, params, xname, barname):

        fiSim = fiSim[:, 0]
        fiSim = np.meshgrid(np.zeros(Fe_eff_grc.shape[1]), fiSim)[1]
        levels = np.unique(fiSim)  # to store for colors

        if P is not None:
            params['P'] = P

        # # #### FIGURE AND COLOR GRADIENT STUFF

        fig1 = plt.figure(figsize=(6, 4))
        plt.subplots_adjust(bottom=.2, left=.15, right=.85, wspace=.2)
        ax = plt.subplot2grid((1, 8), (0, 0), colspan=7)
        ax_cb = plt.subplot2grid((1, 8), (0, 7))

        # -- Setting up a colormap that's a simple transtion
        mymap = plt.cm.viridis
        #mymap = get_linear_colormap()
        build_bar_legend(np.round(levels, 1), levels.size, ax_cb, mymap,
                         label=barname + ' ($\\nu_i$ [Hz])')  # BUILD THE COLOR SCALE BAR

        for i in range(levels.size):
            SIMvector = numTF[i][:]
            SDvector = SDfreq[i][:]
            feSim = Fe_eff_grc[i][:]
            feth = np.linspace(feSim.min(), feSim.max(), int(1e2))
            fi = fiSim[i][0]
            wee = w[i][0]

            _, _, _, muV, sV, muGn, TvN = get_fluct_regime_varsup_eglif(feSim, fi, wee,
                                                                                   *pseq_params_eglif(params),
                                                                                   P[0])

            
            Fout_th = erfc_func(muV, sV, TvN, threshold_func(muV, sV, TvN, muGn, *P), params['Gl'], params['Cm'], alpha)


            cond = Fout_th <= Fout_th.max()

            r = (float(levels[i]) - levels.min()) / (levels.max() - levels.min())

            ax.errorbar(feSim[cond], SIMvector[cond], yerr=SDvector[cond], \
                        color=mymap(r, 1), marker='D', ms=5, capsize=3, elinewidth=1, lw=0)


            ax.plot(feSim[cond], Fout_th[cond], color=mymap(r, 1))


        set_plot(ax, ['bottom', 'left'], xlabel=xname+' ($\\nu_{e}$ [Hz])', \
                 ylabel='$\\nu_{out}$ [Hz]')

        plt.show()

  
        diff = np.absolute(np.array([SIMvector - Fout_th]))
        print('QUANTITATIVE SCORE!!!!!\n diff (Numerical TF - Analytic TF)', diff)


        print('----------------------------------------')
        diff_mean = diff.mean()
        diff_max = diff.max()
        diff_sd = diff.std()
        diff_min = diff.min()
        print('Mean diff +- stdev (numTF-SemiAnalyticTF)', diff_mean, '+-', diff_sd)
        print('Min diff', diff_min)
