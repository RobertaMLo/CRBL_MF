import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from params_library.cell_library import get_neuron_params
from params_library.syn_and_connec_library import get_connectivity_and_synapses_matrix
from params_library.params_reformat import reformat_syn_parameters_eglif, pseq_eglif
from MF_prediction.theoretical_tools import pseq_params, TF_my_template


def load_transfer_functions(CELL, NTWK, P, alpha):
    """
    returns the transfer function of the mean field model
    """

    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, 5)

    # CELL
    params = get_neuron_params(CELL, SI_units=True)
    reformat_syn_parameters_eglif(CELL, params, M)
    print('..................... Loading TFs\nNeuron and Network: ', CELL, NTWK)
    try:
        print('Loaded P: ', P)
        print('\n')
        params['P'] = P

        def TF(fe, fi, XX):
            return TF_my_template(fe, fi, XX, *pseq_params(params), alpha)

    except IOError:
        print('=======================================================')
        print('===========  Fit is not available  ====================')
        print('=======================================================')

    return TF

