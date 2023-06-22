"""
Configurations of the network_scaffold
"""
from __future__ import print_function
import numpy as np


def get_connectivity_and_synapses_matrix(NAME, SI_units=True, number=5):


    # creating empty arry of objects (future dictionnaries)
    M = np.empty((number, number), dtype=object)

    if NAME == 'CRBL_CONFIG_20PARALLEL_wN':
        # MLI AND PC: ALL PARAMS N-NORMALIZED, K PARALLEL TO OTHER POP * 0.2
        ######################### HO MESSO Q * K!!!!!
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        #mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        mf_goc = {'K': 35., 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 374.50, 'Q': 1.126, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 1.244, 'Tsyn': 2.8, 'Erev': -80.}
        pc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}

        # Riempio per colonna - leggo per colonna
        M[:, 0] = [mf_grc.copy(), grc_grc.copy(), goc_grc.copy(), mli_grc.copy(), pc_grc.copy()]  # post-synaptic: grc
        M[:, 1] = [mf_goc.copy(), grc_goc.copy(), goc_goc.copy(), mli_goc.copy(), pc_goc.copy()]  # post-synaptic: goc
        M[:, 2] = [mf_mli.copy(), grc_mli.copy(), goc_mli.copy(), mli_mli.copy(), pc_mli.copy()]  # post-synaptic: mli
        M[:, 3] = [mf_pc.copy(), grc_pc.copy(), goc_pc.copy(), mli_pc.copy(), pc_pc.copy()]  # post-synaptic: pc
        M[:, 4] = [mf_mf.copy(), grc_mf.copy(), goc_mf.copy(), mli_mf.copy(), pc_mf.copy()]  # post-synaptic: mf

        M[0, 0]['name'] = 'to_grc'
        M[0, 1]['name'] = 'to_goc'
        M[0, 2]['name'] = 'to_mli'
        M[0, 3]['name'] = 'to_pc'
        M[0, 4]['name'] = 'to_mf'

 
    else:
        print('====================================================')
        print('------------ NETWORK NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        # quando passo M a load_config.py setto SI_units=True, quindi gli passo i V
        print('synaptic network_scaffold parameters in SI units [V, F, s]')
        for m in M.flatten():
            # NANO SIEMENS ----> SIEMENS
            if 'Q' in m:
                m['Q'] *= 1e-9
            if 'Qp' in m:
                m['Qp'] *= 1e-9
            if 'Qa' in m:
                m['Qa'] *= 1e-9
            # MILLI SEC and MILLI VOLT ----> SEC and VOLT
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3

    else:
        print('synaptic network_scaffold parameters --NOT-- in SI units [mV, pF, ms]')

    return M

if __name__=='__main__':

    M = get_connectivity_and_synapses_matrix('CRBL_CONFIG', 5, SI_units=True)
    print(__doc__)