import argparse
import numpy as np
from CRBL_MF_Model.MF_validation.plot_babe import plot_MF
from CRBL_MF_Model.MF_validation.predictions_analysis import get_auc
import sys

def print_info_sim(sim_vector, namepop):
    print('************************ Scores for: ', namepop,'*************************************')
    if 'BSB-NEST' in namepop:
        dt = 15*1e-3
    elif 'MF' in namepop:
        dt = 1e-4

    print('-- AUC [Hz]: ', get_auc(sim_vector, dt))
    print('-- Peak [Hz]: ', np.max(sim_vector))
    print('-- Avg+-SD [Hz]: ', np.average(sim_vector),'+-',np.std(sim_vector))

    if 'MF' in namepop:
        print('-- Min [Hz]: ', np.min(sim_vector[100:]))
        print('-- baseline at 1ms [Hz]: ',sim_vector[10])

    elif 'BSB-NEST' in namepop:
        print('-- Min [Hz]: ', np.min(sim_vector))


    if namepop == 'PC MF':
        if 'updown' in args.SIM_FILENAME:
            print('-- Pause[Hz]: ', np.min(sim_vector[3600:4000]))
    print('***************************************************************************************')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                                     """ 
                                   Load results mf simulations and print info'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-SIM_FILENAME',
                        help="npy filename (without path) where the scores are saved")

    args = parser.parse_args()

    GrC, GoC, MLI, PC, fmossy = np.load(args.SIM_FILENAME, allow_pickle=True)

    MLIb_psth, GoC_psth, GrC_psth, PC_psth, MLIs_psth = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/'+
                                                                args.SIM_FILENAME.replace('.npy', '_PSTHvals.npy'),
                                                                allow_pickle=True)

    stdoutOrigin = sys.stdout
    sys.stdout = open('/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/'+
                                args.SIM_FILENAME.replace('.npy', '_MFvsBSB.txt'), "w")

    print('INFO FOR: ',args.SIM_FILENAME)

    print_info_sim(GrC, 'GrC MF')
    print_info_sim(GrC_psth, 'GrC BSB-NEST')

    print_info_sim(GoC, 'GoC MF')
    print_info_sim(GoC_psth, 'GoC BSB-NEST')

    print_info_sim(MLI, 'MLI MF')
    print_info_sim(MLIb_psth, 'BASK BSB-NEST')
    print_info_sim(MLIs_psth, 'STELL BSB-NEST')

    print_info_sim(PC, 'PC MF')
    print_info_sim(PC_psth, 'PC BSB-NEST')

    sys.stdout.close()
    sys.stdout = stdoutOrigin

    #t = np.arange(0, 1.5+1e-4, 1e-4)
    #plot_MF(t, [GrC, GoC, MLI, PC, fmossy], 'MFsim', '')
