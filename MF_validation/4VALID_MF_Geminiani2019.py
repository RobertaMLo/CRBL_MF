from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF_2 import find_fixed_point_first_order, find_fixed_point_mossy,\
    plot_for_thesis_activity

from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF import plot_for_thesis_activity_mossy

import numpy as np
import sys
sys.path.append('../')
from CRBL_MF_Model.MF_prediction import *
from CRBL_MF_Model.MF_prediction.load_config_TF import *
from CRBL_MF_Model.MF_prediction.input_library import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

NRN1, NRN2, NRN3, NRN4 = 'GrC', 'GoC', 'MLI', 'PC'
NTWK = 'CRBL_CONFIG_20PARALLEL_wN'

"""
FILE_GrC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_MLI = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
"""

FILE_GrC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_MLI = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'

TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=1.7)
TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.5)
TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=1.5)
TFpc = load_transfer_functions(NRN4, NTWK, FILE_PC, alpha=2.0)

w = 0
dt = 1e-4
t = np.arange(0, 1.77+dt, dt)
T = 3.5e-3

CI_vec1 = [0.5, 10, 8.5, 20]

#json file: background noise
f_backnoise = np.random.rand(len(t)) * 4
#json file: stim on mossy fibers
f_tone = rect_input(time=t, t_start=1000, t_end=1260, minval=0, freq=40, noise_freq=4)
#json file: stim on PC
#f_airpuff = rect_input(time=t, t_start=1260, t_end=1270, minval=0, freq=500, noise_freq=4)

fmossy = f_backnoise + f_tone #+ f_airpuff

X1, fix_grc, fix_goc, fix_mli, fix_pc = find_fixed_point_first_order(TFgrc, TFgoc, TFmli, TFpc,
                                                                         CI_vec1, t, w, fmossy, T)
mytitle = "CRBL MF - 1st Order"
plot_for_thesis_activity_mossy(t, X1, fmossy, mytitle)