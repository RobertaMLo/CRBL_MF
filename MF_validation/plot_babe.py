import numpy as np
from random import gauss, seed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
#import plotly.io as pio



def plot_MF(t, X, mytitle, im_name):
    t = t*1000
    t_max = np.max(t)
    fig = make_subplots(rows=5, cols=1,
                        x_title='time [ms]',
                        y_title='Population activity [Hz]',
                        subplot_titles=('PC', 'MLI', 'GrC', 'GoC', 'Driving Input'))

    fig.add_trace(go.Scatter(x=t, y=X[0], mode='lines', name='$\\nu_{GrC}$',  line=dict(color = "red")),
                      row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[1], mode='lines', name='$\\nu_{GoC}$', line=dict(color = "blue")),
                      row=4, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[2], mode='lines', name='$\\nu_{MLI}$', line=dict(color = "salmon")),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[3], mode='lines', name='$\\nu_{PC}$', line=dict(color = "green")),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[4], mode='lines', name='$\\nu_{mossy}$', line=dict(color = "orchid")),
                      row=5, col=1)



    fig.update_layout(title=mytitle)

    fig.update_layout(
        title=mytitle,
        autosize=True,
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[x_i for x_i in xval],
        #ticktext=[str(int(x_i * 100)) + '%' for x_i in xval],
        # tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle=0,
        title_standoff=25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        #ticktext=[str(round(y_i, 2)) for y_i in yval],
        tickangle=0,
        #title_text=ytitle,
        #title_font={"size": fontsize},
        title_standoff=25)

    fig.show()
    fig.write_html(os.getcwd()+'/imgs/'+im_name+".html")
    fig.write_html(os.getcwd() + '/imgs/' +im_name+ ".svg")

    #pio.write_image(fig, 'imgs/'+im_name, 'png')

def plot_MLI_PC(t, X, mytitle, im_name):

    t = t*1000
    fig = make_subplots(rows=3, cols=1,
                        x_title='time [ms]',
                        y_title='Populaton activity [Hz]',
                        subplot_titles=('Molecular Layer Interneurons', 'Purkinje Cells', 'Driving Input'))


    fig.add_trace(go.Scatter(x=t, y=X[0], mode='lines', name='$\\nu_{MLI}$', line=dict(color = "salmon")),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[1], mode='lines', name='$\\nu_{PC}$', line=dict(color = "green")),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[2], mode='lines', name='$\\nu_{mossy}$', line=dict(color = "orchid")),
                      row=3, col=1)

    #fig['layout']['xaxis5']['title'] = 'time [ms]'
    #fig['layout']['yaxis']['title'] = 'Population Activity [Hz]'


    fig.update_layout(title=mytitle)

    fig.update_layout(
        title=mytitle,
        autosize=True,
        #width=1000,
        #height=600,
        #margin=dict( l=5, r=50, b=1, t=100, pad=4),
        #paper_bgcolor="white",
    )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[x_i for x_i in xval],
        #ticktext=[str(int(x_i * 100)) + '%' for x_i in xval],
        # tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle=0,
        title_standoff=25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        #ticktext=[str(round(y_i, 2)) for y_i in yval],
        tickangle=0,
        #title_text=ytitle,
        #title_font={"size": fontsize},
        title_standoff=25)


    fig.show()
    fig.write_html(os.getcwd()+'/imgs/'+im_name+".html")
    fig.write_html(os.getcwd() + '/imgs/' +im_name+ ".svg")


def plot_PC(t, X, mytitle, im_name):

    t = t*1000
    fig = make_subplots(rows=1, cols=1,
                        x_title='time [ms]',
                        y_title='Populaton activity [Hz]',
                        subplot_titles=('Purkinje Cells')
                        )


    fig.add_trace(go.Scatter(x=t, y=X, mode='lines', name='$\\nu_{MLI}$', line=dict(color = "green")),
                      row=1, col=1)


    fig.update_layout(title=mytitle)

    fig.update_layout(
        title=mytitle,
        autosize=True,
        #width=1000,
        #height=600,
        #margin=dict( l=5, r=50, b=1, t=100, pad=4),
        #paper_bgcolor="white",
    )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[x_i for x_i in xval],
        #ticktext=[str(int(x_i * 100)) + '%' for x_i in xval],
        # tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle=0,
        title_standoff=25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        #ticktext=[str(round(y_i, 2)) for y_i in yval],
        tickangle=0,
        #title_text=ytitle,
        #title_font={"size": fontsize},
        title_standoff=25)


    fig.show()
    fig.write_html(os.getcwd()+'/imgs/'+im_name+".html")
    fig.write_html(os.getcwd() + '/imgs/' +im_name+ ".svg")


def plot_PC_Dyst(t, X, mytitle, im_name):
    t = t*1000
    t_max = np.max(t)
    fig = make_subplots(rows=4, cols=1,
                        x_title='time [ms]',
                        y_title='Activity [Hz]',
                        subplot_titles=('Purkinje Cells','Conditioned Stimulus', 'Unconditioned Stimulus', 'IO aberrrant firing rate'))

    fig.add_trace(go.Scatter(x=t, y=X[0], mode='lines', name='$\\nu_{PC}$',  line=dict(color = "green")),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[1], mode='lines', name='$\\nu_{CS}$', line=dict(color = "orchid")),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[2], mode='lines', name='$\\nu_{US}$', line=dict(color = "indigo")),
                      row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=X[3], mode='lines', name='$\\nu_{IO}$', line=dict(color = "deeppink")),
                      row=4, col=1)

    fig.update_layout(title=mytitle)

    fig.update_layout(
        title=mytitle,
        autosize=True,
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[x_i for x_i in xval],
        #ticktext=[str(int(x_i * 100)) + '%' for x_i in xval],
        # tickvals= [0, 30, 50, 100, 150, 170],#list(range(0, 200, 10)),
        tickangle=0,
        title_standoff=25
    )

    fig.update_yaxes(
        tickfont=dict(family="Arial", size=15),
        tickmode="array",
        #tickvals=[y_i for y_i in yval],
        #ticktext=[str(round(y_i, 2)) for y_i in yval],
        tickangle=0,
        #title_text=ytitle,
        #title_font={"size": fontsize},
        title_standoff=25)

    fig.show()
    fig.write_html(os.getcwd()+'/imgs/'+im_name+".html")
    fig.write_html(os.getcwd() + '/imgs/' +im_name+ ".svg")

    #pio.write_image(fig, 'imgs/'+im_name, 'png')


def updown_protocol():
    dt = 1e-4
    t = np.arange(0, 1.6+dt, dt)

    #X_40 = np.load('updown_MF1ord_40.npy', allow_pickle=True)
    #X_70 = np.load('updown_MF1ord_70.npy', allow_pickle=True)
    #X_100 = np.load('updown_MF1ord_100.npy', allow_pickle=True)

    #X_50 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/new_updown_MF2ord_50.npy')
    X_50 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/old_updown_MF2ord_50.npy')


    plot_MF(t, X_50,  'Up-Down protocol - Input: 50 Hz', 'provaaaa')



def updown_protocol_2ord():
    dt = 1e-4
    t = np.arange(0, 1+dt, dt)

    X_40 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/updown_MF2ord_40.npy', allow_pickle=True)
    X_70 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/updown_MF2ord_70.npy', allow_pickle=True)
    X_100 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/updown_MF2ord_100.npy', allow_pickle=True)

    plot_MF(t, X_40,  'Up-Down protocol - Second Order - Input: 40 Hz', 'updown_MF2ndOrd_40')
    plot_MF(t, X_70,  'Up-Down protocol - Second Order - Input: 70 Hz', 'updown_MF2ndOrd_70')
    plot_MF(t, X_100, 'Up-Down protocol - Second Order - Input: 100 Hz','updown_MF2ndOrd_100')


def updown_protocol_20_desync():
    dt = 1e-4
    t = np.arange(0, 1.6+dt, dt)

    #X_40 = np.load('updown_MF1ord_40.npy', allow_pickle=True)
    X_70 = np.load('updown_20_desync_MF1ord_70.npy', allow_pickle=True)
    #X_100 = np.load('updown_MF1ord_100.npy', allow_pickle=True)

    #plot_MF(t, X_40,  'Up-Down protocol - Input: 40 Hz', 'updown_MF1stOrd_40')
    plot_MF(t, X_70,  'Up-Down desync protocol - Input: 70 Hz', 'updown_20_desync_MF1stOrd_70')
    #plot_MF(t, X_100, 'Up-Down protocol - Input: 100 Hz','updown_MF1stOrd_100')


def updown_protocol_20_desync_2ord():
    dt = 1e-4
    t = np.arange(0, 1.6+dt, dt)

    #X_40 = np.load('updown_MF1ord_40.npy', allow_pickle=True)
    X_70 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/updown_20_desync_MF2ord_70.npy', allow_pickle=True)
    #X_100 = np.load('updown_MF1ord_100.npy', allow_pickle=True)

    #plot_MF(t, X_40,  'Up-Down protocol - Input: 40 Hz', 'updown_MF1stOrd_40')
    plot_MF(t, X_70,  'Up-Down desync protocol - Input: 70 Hz', 'updown_20_desync_MF2ndOrd_70')
    #plot_MF(t, X_100, 'Up-Down protocol - Input: 100 Hz','updown_MF1stOrd_100')


def updown_protocol_short():
    dt = 1e-4
    t = np.arange(0, 0.1+dt, dt)

    X_40 = np.load('updown_short_MF1ord_40.npy', allow_pickle=True)
    X_70 = np.load('updown_short_MF1ord_70.npy', allow_pickle=True)
    X_100 = np.load('updown_short_MF1ord_100.npy', allow_pickle=True)

    plot_MF(t, X_40,  'Short Up-Down protocol - Input: 40 Hz','updown_short_MF1stOrd_40')
    plot_MF(t, X_70,  'Short Up-Down protocol - Input: 70 Hz','updown_short_MF1stOrd_70')
    plot_MF(t, X_100, 'Short Up-Down protocol - Input: 100 Hz','updown_short_MF1stOrd_100')


def updown_protocol_short_2ord():
    dt = 1e-4
    t = np.arange(0, 0.1+dt, dt)

    #X_40 = np.load('updown_short_MF2ord_40.npy', allow_pickle=True)
    X_70 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/updown_short_MF2ord_70.npy', allow_pickle=True)
    #X_100 = np.load('updown_short_M21ord_100.npy', allow_pickle=True)

    #plot_MF(t, X_40,  'Short Up-Down protocol - Input: 40 Hz','updown_short_MF2ndOrd_40')
    plot_MF(t, X_70,  'Short Up-Down protocol - Input: 70 Hz','updown_short_MF2ndOrd_70')
    #plot_MF(t, X_100, 'Short Up-Down protocol - Input: 100 Hz','updown_short_MF1ndOrd_100')


def sin_protocol():
    dt = 1e-4
    t = np.arange(0, 1 + dt, dt)

    X_40 = np.load('sin_MF1ord_40.npy', allow_pickle=True)
    X_70 = np.load('sin_MF1ord_70.npy', allow_pickle=True)
    X_100 = np.load('sin_MF1ord_100.npy', allow_pickle=True)
    X_7_5 = np.load('sin_MF1ord_7.5.npy', allow_pickle=True)
    X_7_5_15 = np.load('sin_MF1ord_15_7_5.npy', allow_pickle=True)
    X_15 = np.load('sin_MF1ord_15.npy', allow_pickle=True)

    plot_MF(t, X_40, 'Syne wave protocol - Input: 40 Hz', 'sin_MF1stOrd_40')
    plot_MF(t, X_70, 'Syne wave protocol - Input: 70 Hz', 'sin_MF1stOrd_70')
    plot_MF(t, X_100, 'Syne waveprotocol - Input: 100 Hz', 'sin_MF1stOrd_100')
    plot_MF(t, X_7_5, 'Syne waveprotocol - Input: 7.5 Hz', 'sin_MF1stOrd_7_5')
    plot_MF(t, X_15, 'Syne waveprotocol - Input: 15 Hz', 'sin_MF1stOrd_15')
    plot_MF(t, X_7_5_15, 'Syne waveprotocol - Input: 7.5 + 15 Hz', 'sin_MF1stOrd_15')


def sin_protocol_2ord():
    dt = 1e-4
    t = np.arange(0, 1 + dt, dt)

    X_15 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/sin_MF2ord_15.npy', allow_pickle=True)
    X_40 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/sin_MF2ord_40.npy', allow_pickle=True)
    X_70 = np.load('/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/sin_MF2ord_70.npy', allow_pickle=True)

    plot_MF(t, X_15, 'Syne wave protocol - 2nd Order - Input: 15 Hz', 'sin_MF2ndOrd_15')
    plot_MF(t, X_40, 'Syne wave protocol - 2nd Order - Input: 40 Hz', 'sin_MF2ndOrd_40')
    plot_MF(t, X_70, 'Syne waveprotocol - 2nd Order - Input: 70 Hz', 'sin_MF2ndOrd_70')


def sin_20_desync_protocol():
    dt = 1e-4
    t = np.arange(0, 1.6 + dt, dt)

    X_40 = np.load('sin_20_desync_MF1ord_40.npy', allow_pickle=True)

    plot_MF(t, X_40, 'Syne wave desync protocol - Input: 40 Hz', 'sin_20_desync_MF1stOrd_40')



def sin_20_desync_protocol_2ord():
    dt = 1e-4
    t = np.arange(0, 1.6 + dt, dt)

    X_40 = np.load('sin_20_desync_MF2ord_40.npy', allow_pickle=True)

    plot_MF(t, X_40, 'Syne wave desync protocol - 2nd Order - Input: 40 Hz', 'sin_20_desync_MF2ndOrd_40')


if __name__ == '__main__':
    updown_protocol()
    sin_20_desync_protocol()
    sin_20_desync_protocol_2ord()
    updown_protocol_20_desync_2ord()
    updown_protocol_20_desync()
    #updown_protocol_short_2ord()
    #sin_protocol_2ord()
    #updown_protocol_2ord()
    #sin_protocol()
    #updown_protocol()
    #updown_protocol_short()