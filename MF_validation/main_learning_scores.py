import numpy as np

from predictions_analysis import plot_dots, load_result_learnig, write_results
from CRBL_MF_Model.MF_prediction.input_library import *
import matplotlib.pyplot as plt
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Evaluate the results of learning insepction:
                                   Load results, compute score (PC AUC and Peak reduction/increase), plot dots'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-SCORE_FILE',
                        help="filename where the scores are saved")

    args = parser.parse_args()

    PF_PCw, MLI_PCw, AUC, Peak, Maximum, Pause = load_result_learnig(args.SCORE_FILE)

    if 'PFPC' in args.SCORE_FILE:
        ind_1 = list(PF_PCw).index(1.0)
        strtoprint = 'Pfs-PCw'
        xval = PF_PCw
        xtitle =  '% of Pf-PC conductance'
        img_prefix = 'PFPC'

    elif 'MLIPC' in args.SCORE_FILE:
        ind_1 = list(MLI_PCw).index(1.0)
        strtoprint = 'MLI-PCw'
        xval = MLI_PCw
        xtitle = '% of MLI-PC conductance'
        img_prefix = 'MLIPC_'

    AUC_norm_1 = AUC/AUC[ind_1]
    Peak_norm_1 = Peak/Peak[ind_1]
    Maximum_norm_1 = Maximum/Maximum[ind_1]
    Pause_norm_1 = Pause / Pause[ind_1]

    print('*********************************************************************\n'
          'Evaluation scores for '+args.SCORE_FILE+'\n'
          '*********************************************************************')

    print('\n\n% of reduction compared to the standard condition = 100%')

    print(strtoprint+'\t\t\t\t % diff AUC\t\t % diff Peak\t\t % diff Max\t\t % diff Pause')

    for i in np.arange(0, len(PF_PCw), 1):

        diff_AUC_i = (AUC_norm_1[i]-AUC_norm_1[ind_1]) * 100
        diff_Peak_i = (Peak_norm_1[i]-Peak_norm_1[ind_1]) * 100
        diff_Max_i = (Maximum_norm_1[i]-Maximum_norm_1[ind_1]) * 100
        diff_Pause_i = (Pause_norm_1[i] - Pause_norm_1[ind_1]) * 100

        if 'MLIPC' in args.SCORE_FILE:

            print('%.2f' % MLI_PCw[i], '-', '%.2f' % MLI_PCw[ind_1],
                  '\t\t', '%.2f' % diff_AUC_i, '\t\t', '%.2f' % diff_Peak_i,
                    '\t\t','%.2f' % diff_Max_i, '\t\t\t','%.2f' % diff_Pause_i)


        elif 'PFPC' in args.SCORE_FILE:
            print('%.2f' % PF_PCw[i], '-', '%.2f' % PF_PCw[ind_1],
                  '\t\t', '%.2f' % diff_AUC_i, '\t\t', '%.2f' % diff_Peak_i,
                    '\t\t','%.2f' % diff_Max_i, '\t\t\t','%.2f' % diff_Pause_i)


        header = ['PF_PCw', 'MLI_PCw', '%diff_AUC', '%diff_Peak', '%diff_Max', '%diff_Pause']
        data = [PF_PCw[i], MLI_PCw[i], diff_AUC_i, diff_Peak_i, diff_Max_i, diff_Pause_i]
        filename = args.SCORE_FILE +'_differences'
        write_results(header, data, filename)


    plot_dots(xval=xval, yval=AUC_norm_1[:-1], legendname='AUC', pltitle='',
              xtitle=xtitle,
              #xtitle='% of Pf-PC conductance',
              ytitle = 'PC AUC',  #ytitle = 'PC AUC norm'
              fontsize=20, name_img=img_prefix)

    plot_dots(xval=xval, yval=Peak_norm_1[:-1], legendname='Peak', pltitle='',
              xtitle=xtitle,
              #xtitle='% of Pf-PC conductance',
              ytitle = 'PC Peak', fontsize=20, name_img=img_prefix)

    plot_dots(xval=xval, yval=Pause_norm_1[:-1], legendname='Peak', pltitle='',
              xtitle=xtitle,
              #xtitle='% of Pf-PC conductance',
              ytitle = 'PC Pause', fontsize=20, name_img=img_prefix+'_limit265_equispaced')

