from predictions_analysis import run_sims, get_auc, routine_plasticity
from CRBL_MF_Model.MF_prediction.input_library import *
import matplotlib.pyplot as plt
import argparse
import time
from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF_2 import plot_MF, plot_MLI_PC

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


    else: print('Input name not valid')

    return fmossy


if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Compute correlation between signals'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("OUT_FILENAME",
                        help="filename to save simulation output")
    parser.add_argument('-input_name', default = 'updown_std',
                        help="input name")
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

    GrC, GoC, MLI, PC, _ ,t = run_sims(args.OUT_FILENAME, fmossy, args.len_sim_sec, wpce=1, wpci=1, dt=args.dt,
                                     fac_alfa_MLI=args.fac_alfa_MLI, T = args.T,
                                    root_path= args.root_numTF, w = args.w)

    date_time = time.strftime("%Y%m%d_%H%M%S")
    plot_MF(t, np.array([GrC,  GrC, GoC, MLI, PC, fmossy]), 'MF activity', args.OUT_FILENAME+'_'+date_time)
    plot_MLI_PC(t, np.array([MLI, PC, fmossy]), 'Molecular Layer activity', args.OUT_FILENAME+'_'+date_time)










