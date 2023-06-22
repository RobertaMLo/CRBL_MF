import seaborn as sns; sns.set()
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def solve_MF_GrC(TF, time, h, T, v_grc0, v_goc, F_mossy, w):
    TF_tot = []
    v_grc = []

    for t in range(len(time)):
        if t == 0:
            v_grc.append(v_grc0)
        else:
            TF_grc_MF = TF(F_mossy[t-1], v_goc[t-1], w[t-1])

            update_v_grc = v_grc[t - 1] + h * (TF_grc_MF - v_grc[t - 1]) / T
            print('update: ', update_v_grc)

            v_grc.append(update_v_grc)

            TF_tot.append(TF_grc_MF)

    return v_grc, TF_tot


def plot_MF_activity(t, v, TF, Fe, Fi):
    fig = go.Figure()

    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(go.Scatter(x=t, y=v,
                             mode='lines',
                             name='vout [Hz]'),
                  row = 1, col = 1

                  )
    fig.add_trace(go.Scatter(x=t, y=TF,
                             mode='lines',
                             name='TF [Hz] '),
                  row = 2, col = 1
                  )

    fig.add_trace(go.Scatter(x=t, y=Fe,
                             mode='lines',
                             name='Fe input [Hz] '),
                  row = 3, col =1
                  )

    fig.add_trace(go.Scatter(x=t, y=Fi,
                             mode='lines',
                             name='Fi input [Hz] '),
                  row = 4, col = 1)

    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    fig.update_xaxes(title_text="time [s]", row=4, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="v_out [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="TF [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="Fe input [Hz]", row=3, col=1)
    fig.update_yaxes(title_text="Fi input [Hz]", row=4, col=1)

    # Update title
    fig.update_layout(title_text='Mean Field Prediction', height = 900)
    fig.show()





