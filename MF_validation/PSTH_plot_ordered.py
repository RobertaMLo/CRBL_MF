import numpy as np
from bsb.plotting import hdf5_plot_psth
from bsb.core import from_hdf5
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import argparse
import os

def MFonPSTH_plot_ordered(filename):

    y_bask, y_goc, y_grc, y_pc, y_stell = np.load(filename, allow_pickle=True)
    y_mli = np.average([y_bask, y_stell], axis=0)
    GrC, GoC, MLI, PC, fmossy = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/'+filename.replace('_PSTHvals.npy','.npy'), allow_pickle=True)

    #GrC, GoC, MLI, PC, fmossy = GrC[0:5000], GoC[0:5000], MLI[0:5000], PC[0:5000], fmossy[0:5000]

    dt = 1e-4
    t = np.arange(0, 0.5 + dt, dt)*1000

    window=[0, 500]
    bw=15
    bins=np.arange(int(window[0]), int(window[1]), bw)

    ypsth = [y_pc, y_mli, y_goc, y_grc]
    ymfm = [PC, MLI, GoC, GrC, fmossy]
    clrs = ['green','salmon','blue','red']
    names = ['$\\nu_{PC}$ MFM','$\\nu_{MLI}$ MFM','$\\nu_{GoC}$ MFM','$\\nu_{GrC}$ MFM']
    names_snn = ['$\\nu_{PC}$ SNN', '$\\nu_{MLI}$ SNN', '$\\nu_{GoC}$ SNN', '$\\nu_{GrC}$ SNN']

    fig = make_subplots(rows=5, cols=1,
                    x_title='time [ms]',
                    y_title='Populaton activity [Hz]')
                    #subplot_titles=('PC', 'MLI', 'GoC', 'GrC', 'Driving Input'))

    for i in np.arange(0,4,1):


        fig.add_trace(go.Scatter(
             x=t,
            y=ymfm[i],
            mode='lines',
            name=names[i]+'MFM',
            line=dict(color=clrs[i])),
            row=i+1, col=1)

        trace = go.Bar(
            x=bins,
            y=ypsth[i],
            name = names_snn[i],
            opacity=0.3,
            marker=dict(color=clrs[i])
        )
        fig.add_trace(trace, row=i+1, col=1)


    fig.add_trace(go.Scatter(
    x = t,
    y=fmossy,
    mode = 'lines',
    name = '$\\nu_{drive}$',
    line = dict(color='orchid')),
    row = 5, col = 1)

    fig.update_layout(
        autosize=True,
        # width=1000,
        # height=600,
        # margin=dict(l=5, r=50, b=1, t=100, pad=4),
        paper_bgcolor="white",
        font=dict(family="Arial", size=10),
    )

    fig.show()
    savefig_title = filename.replace('.npy','')+'_Def'
    fig.write_image(os.getcwd() + '/imgs/' + savefig_title + '.svg', scale=1, width=400, height=400)
    fig.write_html(os.getcwd() + '/imgs/' + savefig_title + '.html')

filename1 = '4paper_updown1_500_PSTHvals.npy'
filename2 = '4paper_syn_500_PSTHvals.npy'
filename3 = '4paper_trysyn_500_PSTHvals.npy'
filename4 = '4paper_updown_trysyn_500_PSTHvals.npy'


#filename4 = '4paper_syn_theta_PSTHvals.npy'
#filename3 = '4paper_trysyn_PSTHvals.npy'


MFonPSTH_plot_ordered(filename1)
MFonPSTH_plot_ordered(filename2)
MFonPSTH_plot_ordered(filename3)
MFonPSTH_plot_ordered(filename4)
