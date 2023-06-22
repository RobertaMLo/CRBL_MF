import seaborn as sns; sns.set()
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def solve_MF_GrC_GoC(TF_grc, TF_goc, time, h, T, v_grc0, v_goc0, F_mossy, w):
    TF_tot_grc = []
    TF_tot_goc = []
    v_grc = []
    v_goc = []

    for t in range(len(time)):
        if t == 0:
            v_grc.append(v_grc0)
            v_goc.append(v_goc0)
            TF_tot_grc.append(v_grc0)
            TF_tot_goc.append(v_goc0)
        else:
            TF_grc_MF = TF_grc(F_mossy[t-1], v_goc[t-1], w[t-1])
            TF_goc_MF = TF_goc(F_mossy[t-1], v_grc[t-1], v_goc[t-1], w[t-1])

            update_v_grc = v_grc[t - 1] + h * (TF_grc_MF - v_grc[t - 1])/ T
            update_v_goc = v_goc[t - 1] + h * (TF_goc_MF - v_goc[t - 1])/ T
            #print('update granule: ', update_v_grc)
            #print('update golgi: ', update_v_goc)

            v_grc.append(update_v_grc)
            v_goc.append(update_v_goc)

            TF_tot_grc.append(TF_grc_MF)
            TF_tot_goc.append(TF_goc_MF)

    return v_grc, v_goc, TF_tot_grc, TF_tot_goc


def plot_MF_activity(t, v_grc, v_goc, TF_grc, TF_goc, Fe):
    fig = go.Figure()

    fig = make_subplots(rows=5, cols=1)

    fig.add_trace(go.Scatter(x=t, y=v_grc,
                             mode='lines',
                             name='vout GrC [Hz]'),
                  row = 1, col = 1

                  )
    fig.add_trace(go.Scatter(x=t, y=v_goc,
                             mode='lines',
                             name='vout GoC [Hz]'),
                  row = 2, col = 1

                  )
    fig.add_trace(go.Scatter(x=t, y=TF_grc,
                             mode='lines',
                             name='TF GrC [Hz] '),
                  row = 3, col = 1
                  )

    fig.add_trace(go.Scatter(x=t, y=TF_goc,
                             mode='lines',
                             name='TF GoC [Hz] '),
                  row = 4, col = 1
                  )

    fig.add_trace(go.Scatter(x=t, y=Fe,
                             mode='lines',
                             name='Fe input [Hz] '),
                  row = 5, col =1
                  )

    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    fig.update_xaxes(title_text="time [s]", row=4, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="GrC activity [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="GoC activity [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="TF GrC [Hz]", row=3, col=1)
    fig.update_yaxes(title_text="TF GoC [Hz]", row=4, col=1)
    fig.update_yaxes(title_text="mossy fibers [Hz]", row=5, col=1)

    # Update title
    fig.update_layout(title_text='Mean Field Prediction', height = 900)
    fig.show()





