from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF_2 import find_fixed_point_first_order, find_fixed_point_mossy,\
    plot_for_thesis_activity

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
from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF_2 import *

NRN1, NRN2, NRN3, NRN4 = 'GrC', 'GoC', 'MLI', 'PC'
NTWK = 'CRBL_CONFIG_20PARALLEL_wN'

"""
FILE_GrC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_MLI = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
"""

"""
FILE_GrC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_MLI = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'

TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=1.7)
TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.5)
TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=1.5)
TFpc = load_transfer_functions(NRN4, NTWK, FILE_PC, alpha=2.0)
"""

root_path = '/home/bcc/bsb-ws/CRBL_MF_Model/'
NRN1, NRN2, NRN3, NRN4 = 'GrC', 'GoC', 'MLI', 'PC'

NTWK = 'CRBL_CONFIG_20PARALLEL_wN'
"""
FILE_GrC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_MLI = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
"""

FILE_GrC = root_path + '20220519_120033_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
FILE_GoC = root_path + '20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
FILE_MLI = root_path + '20220519_120011_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.6_fit.npy'
FILE_PC = root_path + '20220519_120128_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.9_fit.npy'

TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=2.)
TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.3)
TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=1.6)
TFpc = load_transfer_functions(NRN4, NTWK, FILE_PC, alpha=1.9)


# OLD CONFIG
"""
FILE_GrC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_124105_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.7_fit.npy'
FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_220421_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_GoC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220422_202248_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.0_fit.npy'
FILE_MLI = '/home/bcc/bsb-ws/CRBL_MF_Model/20220301_171544_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.5_fit.npy'
FILE_PC = '/home/bcc/bsb-ws/CRBL_MF_Model/20220302_125715_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'

# ALPHA OF OLD CONFIG
TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=1.7)
TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.)
TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=1.5)
TFpc = load_transfer_functions(NRN4, NTWK, FILE_PC, alpha=2.)
"""
w = 0
dt = 1e-4

T = 3.5e-3

CI_vec1 = [0.5, 10, 8.5, 12]

#json file: stim on mossy fibers
#t = np.arange(0, 0.1+dt, dt)
#f_tone1 = rect_input(time=t, t_start=50, t_end=250, minval=0, freq=40, noise_freq=4)
#f_tone2 = rect_input(time=t, t_start=300, t_end=500, minval=0, freq=40, noise_freq=4)
#f_tone3 = rect_input(time=t, t_start=550, t_end=750, minval=0, freq=40, noise_freq=4)

#t = np.arange(0, 1+dt, dt)
#f_tone1 = rect_input(time=t, t_start=500, t_end=3000, minval=0, freq=70, noise_freq=4)
#f_tone2 = rect_input(time=t, t_start=3500, t_end=6000, minval=0, freq=70, noise_freq=4)
#f_tone3 = rect_input(time=t, t_start=6500, t_end=9000, minval=0, freq=70, noise_freq=4)

t = np.arange(0, 1.6+dt, dt)
f_tone1 = rect_input(time=t, t_start=3500, t_end=6000, minval=0, freq=70, noise_freq=4)
f_tone2 = rect_input(time=t, t_start=6500, t_end=9000, minval=0, freq=70, noise_freq=4)
f_tone3 = rect_input(time=t, t_start=9500, t_end=12000, minval=0, freq=70, noise_freq=4)

#json file: background noise
f_backnoise = np.random.rand(len(t)) * 4

fmossy = f_backnoise + f_tone1 + f_tone2 + f_tone3

freq, amplitude = 10, 40
#amplitude_bis = 7.5
#f_sin = amplitude*np.sin(2*np.pi*freq*t)+amplitude #add amplitude value to avoid negative values
#f_sin[:3000] = 0
##f_sin_bis = amplitude_bis*np.sin(2*np.pi*freq*t)+amplitude_bis
#fmossy = f_sin + f_backnoise #+ f_sin_bis


#X1, fix_grc, fix_goc, fix_mli, fix_pc = find_fixed_point_first_order(TFgrc, TFgoc, TFmli, TFpc,
#                                                                         CI_vec1, t, w, fmossy, T)

#np.save('sin_20_desync_MF1ord_40.npy', [X1[:,0], X1[:,1], X1[:,2], X1[:,3], fmossy])
#mytitle1 = "CRBL MF - 1st Order"



"""
Ngrc = 28615
Ngoc = 70
Nmossy = 2336
Nmli = 299
Npc = 99
"""

Ngrc = 28615
Ngoc = 70
Nmossy = 2336
Nmli = 299+147
Npc = 99

CI_vec2 = [1, 30, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0], 12, 15, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]



X = find_fixed_point_mossy(TFgrc, TFgoc, TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Ngoc,
                           Nmossy, Nmli, Npc, T, verbose=False)


# SALVO COSI' PER PLOT!!!!!!!!!! mossy alla fineeeeeee
#np.save('new_CI_updown_20_desync_MF2ord_70.npy', [X[:, 0], X[:, 1], X[:, 9], X[:, 10], X[:, 8]])


mytitle2 = 'CRBL MF - 2nd Order'
plot_for_thesis_activity(t, X, fmossy, mytitle2)
