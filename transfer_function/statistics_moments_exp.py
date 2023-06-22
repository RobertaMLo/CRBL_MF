import numpy as np

def get_fluct_regime_varsup_eglif(fe, fi,
                                  Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki):


    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe, muGi = Qe * Ke * Te * fe, Qi * Ki * Ti * fi
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe + muGi
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (muGe * Ee + muGi * Ei + Gl * El) / muG  #XX = adaptation
    muGn, Tm = muG / Gl, Cm / muG #normalization

    Ue, Ui = Qe / muG * (Ee - muV), Qi / muG * (Ei - muV)

    #sV = np.sqrt( \
    #    fe * (Ue * Te) ** 2 / 2. / (Te + Tm) + \
    #    fi * (Ti * Ui) ** 2 / 2. / (Ti + Tm))
    sV = np.sqrt( \
        fe * Ke * (Ue * Te) ** 2 / 2. / (Te + Tm) + \
        fi * Ki * (Ti * Ui) ** 2 / 2. / (Ti + Tm))

    fe, fi = fe + 1e-9, fi + 1e-9 # just to insure a non zero division

    Tv = (fe * Ke * (Ue * Te) ** 2 + fi * Ki *(Ti * Ui) ** 2) / (
                fe * Ke * (Ue * Te) ** 2 / (Te + Tm) + fi * Ki * (Ti * Ui) ** 2 / (Ti + Tm))
    TvN = Tv * Gl / Cm #normalization

    return muGe, muGi, muG, muV, sV + 1e-12, muGn, TvN, Tv
