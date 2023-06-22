import numpy as np

from predictions_analysis import run_sims, get_auc, routine_plasticity, load_result_learnig
from CRBL_MF_Model.MF_prediction.input_library import *
import matplotlib.pyplot as plt
import argparse

def dispatch_input(input_name, len_sim_sec, dt=1e-4):
    t = np.arange(0, len_sim_sec + dt, dt)
    f_backnoise = np.random.rand(len(t)) * 4

    if input_name == 'updown_std':
        t_start = 1500
        f_tone1 = rect_input(time=t, t_start=t_start, t_end=3500, minval=0, freq=50, noise_freq=0)
        fmossy = f_backnoise + f_tone1

    elif input_name == 'syn_theta':
        freq = 6
        A = 10
        fmossy = f_backnoise + A*np.sin(2*np.pi*freq*t)+A

    else:
        print('Input name not valid')

    return fmossy


if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Compute correlation between signals'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-write_res", type=bool, default= True,
                        help="write or not the output scores")
    parser.add_argument('-input_name', default = 'updown_std',
                        help="name of the input type")
    parser.add_argument('-type_syn', default = 'PFPC',
                        help="synapses to investigate plasticity. Choose MLIPC or PFPC")
    parser.add_argument('-root_numTF', default='/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/',
                        help='protocol directory of numTFs')
    parser.add_argument('-len_sim_sec', type=float, default = 0.5,
                        help="simulation length in seconds")
    parser.add_argument('-dt', type=float, default = 1e-4,
                        help= "time step in seconds")
    parser.add_argument('-T', type=float, default = 3.5e-3,
                        help="Mean Field Time Constant in seconds")
    parser.add_argument('-fac_alfa_MLI', type=float, default=1,
                        help="weight of MLI Inhibition on PC")
    parser.add_argument('-w', type=float, default=0,
                        help="Adaptation")

    args = parser.parse_args()

    fmossy = dispatch_input(args.input_name, args.len_sim_sec)

    wpce_arr = np.array([1])
    wpci_arr = np.array([1])

    if args.type_syn == 'PFPC':
        print('I am simulate plasticity of Pfs-PC synapses, i.e. PC LEARNING!!!!!')
        #wpce_arr = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.35, 1.50, 1.65, 1.80, 2, 2.20, 2.35, 2.50, 2.65, 2.80])
        wpce_arr = np.array([2.20])
    elif args.type_syn == 'MLIPC':
        print('I am simulate plasticity of MLI-PC synapses, i.e. MLI FEED FORWARD INHIB!!!!!')
        wpci_arr = np.array([0.05, 0.30, 0.50, 1, 1.50, 2, 2.50, 3, 3.50])
    else: print('type_syn not valid!!! I am running a standard sim')


    routine_plasticity(args.type_syn, wpci_arr, wpce_arr, fmossy, args.len_sim_sec, args.dt, args.fac_alfa_MLI,
                       args.T, args.root_numTF, args.w, write_res=args.write_res)

