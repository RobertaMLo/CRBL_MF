import numpy as np
import time
from scipy.optimize import minimize
from scipy import signal
from scipy import integrate
import scipy
from CRBL_MF_Model.MF_prediction.master_equation_CRBL_MF_2 import find_fixed_point_mossy, build_up_differential_operator_mossy, \
    plot_MF, plot_MLI_PC, plot_PC
from CRBL_MF_Model.MF_prediction.load_config_TF import *
from CRBL_MF_Model.MF_prediction.input_library import *
import csv
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def find_fixed_point_mossy_4pred(TFgrc, TFgoc, TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Ngoc, Nmossy, Nmli, Npc, T,
                           verbose=False):

    # V = [Vgrc, Vgoc, Cgrcgrc, Cgrcgoc, Cgrcm, Cmgoc, Cgocgoc, Cmm, Vm, Vmli, Vpc, Cmlimli, Cmlipc, Cgrcpc, Cgrcmli,
    #     Cpcpc, Cmligoc, Cmlimossy, Cpcgoc, Cpcmossy]

    ### SECOND ORDER ###

    # simple euler

    X = CI_vec2
    X_vett = np.zeros((len(t), 20))
    for i in range(len(t)):
        last_X = X
        X = X + (t[1] - t[0]) * build_up_differential_operator_mossy(TFgrc, TFgoc, TFmli, TFpc, w,
                                                                Ngrc=Ngrc, Ngoc=Ngoc, Nmossy=Nmossy, Nmli=Nmli, Npc=Npc,
                                                                T=T)(X)
        #X[8] = fmossy[i]/T
        X[8] = fmossy[i]
        #X[9], X[11], X[12], X[14], X[16], X[17] = X[9]*factMLI, X[11]*factMLI, X[12]*factMLI, X[14]*factMLI, \
        #                                          X[16]*factMLI, X[17]*factMLI
        X_vett[i, :] = X

    print('Make sure that those two values are similar !!')
    print('X: ', X)
    print('last X: ', last_X)

    if verbose:
        print(X)
    if verbose:
        print('first order prediction: ', X[-1])

    # return X[-1][0], X[-1][1], np.sqrt(X[-1][2]), np.sqrt(X[-1][3]), np.sqrt(X[-1][4])
    # return X, X[0], X[1], np.sqrt(X[2]), np.sqrt(X[3]), np.sqrt(X[4])
    #return X_vett, X, X[0], X[1], np.sqrt(X[2]), np.sqrt(X[3]), np.sqrt(X[4])
    return X_vett

def compute_cross_corr(sig1, sig2, title):
    corr_vect = signal.correlate(sig1, sig2, 'full','fft')
    corr_vect=corr_vect/np.max(corr_vect)

    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.arange(len(corr_vect)),
        y=corr_vect,
        name='Correlation',
    ))
    fig.show()
    """

    fig = go.Figure(data=go.Scatter(x=np.arange(len(corr_vect)), y=corr_vect,
                                    mode= 'lines+markers',
                                    name='correlation',
                                    text='',
                                    textposition="bottom center",
                                    line=dict(
                                        color="deeppink",
                                        width=3
                                    )
                                    )
                    )

    fig.update_xaxes(
        tickangle=0,
        title_text='',
        title_font={"size": 25},
        title_standoff=25)

    fig.update_yaxes(
        title_text= 'correlation',
        title_font={"size": 25},
        title_standoff=25)

    fig.update_layout(
        title='Correlation between '+title,
        title_font={"size": 25},
        autosize=True,
        # width=1000,
        # height=600,
        # margin=dict(l=5, r=50, b=1, t=100, pad=4),
        paper_bgcolor="white",
        xaxis=dict(
            tickmode='array',
            showticklabels=False
        )
    )

    fig.show()

    return corr_vect

def get_auc(Signal, dt):
    shift = np.arange(len(Signal))*dt
    return integrate.cumtrapz(Signal, shift)[-1]

def set_Ncells():
    Ngrc = 28615
    Ngoc = 70
    Nmossy = 2336
    Nmli = 299 + 147
    Npc = 99
    return Ngrc, Ngoc, Nmossy, Nmli, Npc

def compute_PSD(Signal_mfm, sampl_freq):
    (f, S) = scipy.signal.welch(Signal_mfm, sampl_freq)
    plt.semilogy(f, S)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.title('Power Spectrum Density')
    plt.show()


def run_sims(filename, fmossy, len_sim_sec, wpce, wpci, dt = 1e-4, fac_alfa_MLI=1, T =3.5e-3,
             root_path='/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/', w=0):

    NRN1, NRN2, NRN3, NRN4 = 'GrC', 'GoC', 'MLI', 'PC'

    NTWK = 'CRBL_CONFIG_20PARALLEL_wN'

    FILE_GrC = root_path + '20220519_120033_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    FILE_GoC = root_path + '20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
    FILE_MLI = root_path + '20220622_085550_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.8_fit.npy'
    FILE_PC = root_path + '20220622_085610_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.5_fit.npy'

    TFgrc = load_transfer_functions(NRN1, NTWK, FILE_GrC, alpha=2.0)
    TFgoc = load_transfer_functions_goc(NRN2, NTWK, FILE_GoC, alpha=1.3)
    TFmli = load_transfer_functions(NRN3, NTWK, FILE_MLI, alpha=5 * fac_alfa_MLI)
    TFpc = load_transfer_functions_molt(NRN4, NTWK, FILE_PC, alpha=5, we=wpce, wi=wpci)

    CI_vec2 = [0.5, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, fmossy[0], 8.5, 20, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    Ngrc = 28615
    Ngoc = 70
    Nmossy = 2336
    Nmli = 299 + 147
    Npc = 99

    t = np.arange(0, len_sim_sec+dt, dt)

    X = find_fixed_point_mossy_4pred(TFgrc, TFgoc, TFmli, TFpc, CI_vec2, t, w, fmossy, Ngrc, Ngoc, Nmossy, Nmli, Npc, T,
                                     verbose=False)




    np.save(filename +'_t'+str(len_sim_sec)+'.npy',  [X[:,0], X[:,1], X[:,9], X[:,10], X[:,8]])

    return X[:, 0], X[:, 1], X[:, 9], X[:, 10], X[:, 8], t


def write_results(header, data, filename):
    path = os.getcwd()

    #'a' = append mode to add a new line, 'w' = write mode to rewrite csv file
    with open(filename+'.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        if not os.path.exists(path + filename+'.csv'):
            writer.writerow(header)
            print('File created')
        writer.writerow(data)

        f.close()


def load_result_learning(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    rows = []

    for row in csvreader:
        rows.append(row)

    file.close()

    rows=np.array(rows)

    Pf_PCw = rows[:,0]
    MLI_PCw = rows[:,1]
    AUC = rows[:,2]
    Peak = rows[:, 3]
    Maximum = rows[:, 4]
    Min_pause = rows[:, 5]

    #type string --> need to map into an array of type float
    Pf_PCw = list(map(float, Pf_PCw[1:]))
    Pf_MLIw = list(map(float, MLI_PCw[1:]))
    AUC = list(map(float, AUC[1:]))
    Peak = list(map(float, Peak[1:]))
    Maximum = list(map(float, Maximum[1:]))
    Min_pause = list(map(float, Min_pause[1:]))

    return np.array(Pf_PCw), np.array(Pf_MLIw), np.array(AUC), np.array(Peak), np.array(Maximum), np.array(Min_pause)


def plot_dots(xval, yval, legendname, pltitle, xtitle, ytitle, fontsize, name_img):

    fig = go.Figure(data=go.Scatter(x=xval, y=yval,
                                    mode='markers', #lines+markers
                                    name = legendname,
                                    #text = dot_label,
                                    textposition="bottom center",
                                    marker=dict(
                                        #color = "deeppink",
                                        color = 'green',
                                        size = 20,
                                        line = dict(
                                            color='green',
                                            width=2
                                            )
                                        )
                                    )
                        )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=fontsize),
        tickmode="array",
        tickvals = [x_i for x_i in xval],  # list(range(0, 200, 10)),
        ticktext = [str(int(x_i*100)) for x_i in xval],
        #tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle = 0,
        title_text = xtitle,
        title_font = {"size": fontsize+4},
        title_standoff = 25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=fontsize),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        tickvals= [yval[0], yval[list(yval).index(1.0)], yval[-1]],
        ticktext=[str(round(yval[0],2)), str(round(yval[list(yval).index(1.0)],2)), str(round(yval[-1],2))],
        tickangle=0,
        title_text=ytitle,
        title_font={"size": fontsize+4},
        title_standoff=25)

    fig.update_layout(
        title=pltitle,
        title_font={"size": fontsize+4},
        autosize=True,
        #width=1000,
        #height=600,
        #margin=dict(l=5, r=50, b=1, t=100, pad=4),
        paper_bgcolor="white"
    )

    fig.show()
    fig.write_html(os.getcwd()+'/imgs/'+name_img+ytitle+".html")
    fig.write_html(os.getcwd() + '/imgs/' +name_img+ytitle + ".svg")



def routine_eval_sims(filename):

    Pf_PCw, MLIw, AUC, Peak, Minim, Min_pause = load_result_learning(filename)

    all_scores = np.array([MLIw, AUC, Peak, Minim, Min_pause]).T #Transpose for sorting according to MLIw
    all_scores_ord= all_scores[np.argsort(all_scores[:,0])] #sorted array. argsort will return the indeces
    MLIw, AUC, Peak, Minim, Min_pause = all_scores_ord[:,0], all_scores_ord[:,1], all_scores_ord[:,2], \
                                        all_scores_ord[:,3], all_scores_ord[:,4]

    Diff_pp = Peak - Min_pause
    AUC_norm = AUC/np.max(AUC) #Normalization with maximum
    Peak_norm = Peak/np.max(Peak)
    Min_pause_norm = Min_pause/np.max(Min_pause)

    print('AUC Decrease:\n100%-50%: ',AUC_norm[2]-AUC_norm[0],'\n100%-170%: ',AUC_norm[0]-AUC_norm[-1])
    print('Peak Decrease:\n100%-50%: ', Peak_norm[2] - Peak_norm[0], '\n100%-170%: ', Peak_norm[0] - Peak_norm[-1])

    """
    #AUC Plot
    plot_dots(xval=MLIw*100, yval=AUC, legendname='AUC',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% MLI activity', ytitle='PC AUC', fontsize=25)
    #Peak Plot
    plot_dots(xval=MLIw*100, yval=Peak, legendname='Peak',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% MLI activity', ytitle='PC Peak', fontsize=25)

    #Minimum Plot
    
    plot_dots(xval=MLIw*100, yval=Minim, legendname='Minimum',
              dot_label=[str(MLIw[0]*100)+'%', str(MLIw[1]*100)+'%', str(MLIw[2]*100)+'%', str(MLIw[3]*100)+'%',
                                                                       str(MLIw[4]*100)+'%', str(MLIw[5]*100)+'%'],
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% MLI activity', ytitle='PC Minimum', fontsize=25)
    

    plot_dots(xval=Min_pause, yval=MLIw * 100, legendname='Minimum',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% of active MLIs', ytitle='PC Pause', fontsize=25)
    """

    plot_dots(xval=Diff_pp, yval=MLIw * 100, legendname='Minimum',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% of active MLIs', ytitle='PC Difference Peak-Pause', fontsize=25)

    #Normalised plot
    plot_dots(xval=MLIw*100, yval=AUC_norm, legendname='Minimum',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% of active MLIs', ytitle='Normalized PC AUC', fontsize=25)

    plot_dots(xval=MLIw*100, yval=Peak_norm, legendname='Minimum',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% of active MLIs', ytitle='Normalized PC Peak', fontsize=25)

    plot_dots(xval=MLIw*100, yval=Min_pause_norm, legendname='Minimum',
              pltitle='Evaluation of inhibition impact on the output',
              xtitle='% of active MLIs', ytitle='Normalized PC Pause', fontsize=25)


    return MLIw, AUC, Peak, Minim, Min_pause


def routine_plasticity(type_syn, wpci_arr, wpce_arr, fmossy, len_sim_sec, dt, fac_alfa_MLI, T, root_path, w, write_res):
    date_time = time.strftime("%Y%m%d_%H%M%S")
    OUT_FILENAME = date_time+'_'+type_syn+'_plasticity'

    header = ['w_pf_pc', 'w_mli_pc', 'AUC', 'PEAK', 'Maximum', 'Pause']

    for wpce in wpce_arr:
        for wpci in wpci_arr:

            print('-------------------------------------------Parallel fiberes - Purkinje Cell weight: ', wpce)

            GrC, GoC, MLI, PC, _ , t = run_sims(OUT_FILENAME, fmossy, len_sim_sec, wpce, wpci, dt,
                                                fac_alfa_MLI, T, root_path, w)


            plot_MLI_PC(t, np.array([MLI, PC, fmossy]),
                         'Molecular Layer Activity - Pf-PCw: ' + str(wpce) +' MLI-PCw: ' + str(wpci),
                          date_time +'_PFPC' + str(wpce)+'_MLIPC'+str(wpci))

            #plot_PC(t, np.array([PC]),
            #         'PC Activity - PF-PCw: '+str(wpce), date_time+'_'+str(wpce))

            AUC_PC = get_auc(PC, dt)
            PC_peak = np.max(PC[1400:1600])
            PC_max = np.max(PC)
            PC_pause = np.min(PC[1500 + 50 : 1500+ 1000])

            print('\nAUC: ', AUC_PC)

            print('\nPeak: ', PC_peak)

            data = [wpce, wpci, AUC_PC, PC_peak, PC_max, PC_pause]

            if write_res:
                write_results(header, data, filename=OUT_FILENAME+'_scores')

        print('I finish babyyyy')

