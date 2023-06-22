"""
Configurations of the network_scaffold
"""
from __future__ import print_function
import numpy as np


def get_connectivity_and_synapses_matrix(NAME, SI_units=True, number=5):


    # creating empty arry of objects (future dictionnaries)
    M = np.empty((number, number), dtype=object)

    if NAME=='CRBL_CONFIG': # K = MEAN CONV; Q = Q_unit * num_syns
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.543, 'Tsyn': 1.25, 'Erev': 0.} ## CORRETTO Q! Prima era 0.35
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.132, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 28.3, 'Q': 0.549, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 374.50, 'Q': 0.571, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 25.60, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.}
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


    elif NAME=='CRBL_CONFIG_ALL_PARALLEL': # K = MEAN CONV; Q = Q_unit * num_syns
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.543, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.132, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 28.3, 'Q': 0.549, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.245, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 25.60, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.}
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


    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_Q_NOK':  # K = MEAN CONV; Q = Q_unit * num_syns
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # goc_grc = {'K': 3.5, 'Q': 0.24, 'Tsyn': 4.5, 'Erev': -80.} #old version
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.28, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 28.3, 'Q': 1.1, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 2.252, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 25.60, 'Q': 2.08, 'Tsyn': 2.8, 'Erev': -80.}
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

    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_QMLI_wN':
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 28.3, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.245, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 25.60, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.}
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


    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_KMLIwN_QMLIwN':
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.245, 'Tsyn': 1.1, 'Erev': 0.}
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


    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_KMLIwN_QK': #from mli: K wmean on bas and stell, Q wmean on Kstell and Kbas
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.245, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.}#!!!!!!!!!!! qqqqqqq
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

    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_KMLIwN_QK_QgrcnoK':
        #from mli: K wmean on bas and stell, Q wmean on Kstell and Kbas
        #from grc: K = Kp+Ka, Q = Qp*nsynp+Qa*nsyna
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 2.252, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.}
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


    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_KMLIwN_QmlipcK_QgrcpcMean':
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.510, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 0.693, 'Tsyn': 2.8, 'Erev': -80.} #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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


    elif NAME == 'CRBL_CONFIG_ALL_PARALLEL_MYBESTONFIG':
        # pops param
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        ## new to_grc: K = mean conv, Q = Q*num_syn*K/K
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.35, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 0.510, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 1.24, 'Tsyn': 2.8, 'Erev': -80.}
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


    elif NAME == 'CRBL_CONFIG_20PARALLEL_wN_qk':
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
        grc_goc = {'K': 501.98, 'Q': 0.54, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.13, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.53, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 374.50, 'Q': 0.57, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 0.86, 'Tsyn': 2.8, 'Erev': -80.}
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

    elif NAME == 'CRBL_CONFIG_20PARALLEL_wN':
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

    elif NAME == 'CRBL_CONFIG_20PARALLEL_wN_redGoCGrC':
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
        goc_grc = {'K': 2.5*0.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        # mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        mf_goc = {'K': 35., 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.54, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2*2, 'Q': 1.12, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.13, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.53, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 374.50, 'Q': 0.57, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 0.86, 'Tsyn': 2.8, 'Erev': -80.}
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

    elif NAME == 'CRBL_CONFIG_20_mfs':
        # MLI AND PC: ALL PARAMS N-NORMALIZED, K mfs TO OTHER POP * 0.2
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        mf_grc = {'K': 0.80, 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 2.5, 'Q': 0.336, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 11.42, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 1228.3, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1544.90, 'Q': 1.126, 'Tsyn': 1.1, 'Erev': 0.}
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


    elif NAME == 'FIX_MLI_TO_PC':
        # MLI AND PC: ALL PARAMS N-NORMALIZED, K PARALLEL TO OTHER POP * 0.2
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
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5, 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5, 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        #mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        mli_mli = {'K': 14*99.9+14.3*99.9, 'Q': 0.532, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1463., 'Q': 1.126, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28*0.1, 'Q': 1.24*0.1, 'Tsyn': 2.8, 'Erev': -80.}
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

    elif NAME == 'wN_20PARALLEL_KNsyn':
        # MLI AND PC: ALL PARAMS N-NORMALIZED, K PARALLEL TO OTHER POP * 0.2
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 3.5, 'Q': 0.240, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 2592.00, 'Q': 0.007, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 1418.69, 'Q': 0.005, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 489.16, 'Q': 0.510, 'Tsyn': 1.1, 'Erev': 0.}
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

    elif NAME == 'wN_ALLPARALLEL_KNsyn':
        # MLI AND PC: ALL PARAMS N-NORMALIZED, K PARALLEL TO OTHER POP * 0.2
        # to mf -------------------------------------------------------
        mf_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_mf = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to grc -------------------------------------------------------
        mf_grc = {'K': 4., 'Q': 0.23, 'Tsyn': 1.9, 'Erev': 0.}
        grc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        goc_grc = {'K': 3.5, 'Q': 0.240, 'Tsyn': 4.5, 'Erev': -80.}
        mli_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_grc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to goc Kgrc = Kp*factor + Ka ; Qgrc = (Qa*n_syn_a*Ka+Qp*n_syn_p*Kp*factor)/(Ka+Kp*factor)-------------------------------------------------------
        mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 1228.3, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 2592.00, 'Q': 0.007, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 1219.8, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 1418.69, 'Q': 0.005, 'Tsyn': 2., 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 1659.56, 'Q': 0.510, 'Tsyn': 1.1, 'Erev': 0.}
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

    elif NAME == 'PROVA_TAU':
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
        # mf_goc = {'K': 57.1, 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        mf_goc = {'K': 35., 'Q': 0.24, 'Tsyn': 5., 'Erev': 0.}
        grc_goc = {'K': 501.98, 'Q': 0.437, 'Tsyn': 1.25, 'Erev': 0.}
        goc_goc = {'K': 16.2, 'Q': 1.12, 'Tsyn': 5., 'Erev': -80.}
        mli_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        pc_goc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to mli -----------------------------------------------------------------------
        mf_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_mli = {'K': 243.96, 'Q': 0.154, 'Tsyn': 0.64, 'Erev': 0.}
        goc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_mli = {'K': 14.20, 'Q': 0.532, 'Tsyn': 2, 'Erev': -80.}
        pc_mli = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        # to pc -------------------------------------------------------------------------
        mf_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        grc_pc = {'K': 374.50, 'Q': 1.126, 'Tsyn': 1.1, 'Erev': 0.}
        goc_pc = {'K': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.}
        mli_pc = {'K': 10.28, 'Q': 1.244, 'Tsyn': 1., 'Erev': -80.}
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