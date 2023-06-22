import numpy as np
from bsb.plotting import hdf5_plot_psth
from bsb.core import from_hdf5
import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse
import os


if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   PSTH and MF plots
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-bsb_rec_file',
                        help="filename where the bsb-nest results had been saved")

    parser.add_argument('-type_img', type=int, default=2,
                        help="1 for PSTH, 2 for PSTH and MF overlapped")

    parser.add_argument('-save_imgs', type=bool, default=True,
                        help="decided to save or not image (True or False)")

    args = parser.parse_args()


    path = ''
    network_file = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/balanced.hdf5' #Network structure

    recordings_file = args.bsb_rec_file


    network_scaffold = from_hdf5(network_file)
    adapter = network_scaffold.create_adapter("stim_on_MFs")
    cells = network_scaffold.get_cell_types()
    print([cell.name for cell in cells])

    cutoff = 0             # [ms]
    duration = 1.5

    #METTERE CODICE ESA DI COLORE DA PLOT MF master equation
    color = {'granule_cell': '#62DACF',
         'golgi_cell': '#7E8ACA',
         'purkinje_cell': 'orchid',
         'stellate_cell': 'salmon',
         'basket_cell': 'salmon',
         'glomerulus': '#D11D12',
         #'ubc': network_scaffold.configuration.cell_types["ubc"].plotting.color,
        }
    print("color", color)


    # Mean rate as population firing rate
    results = h5py.File(path+recordings_file, "r")

    mean_rate = {}
    rate = {}

    #Loop on FR values
    for g in results["/recorders/soma_spikes"].values():
        print(g.attrs["label"])
    #Loop on cell type
    for cell_type in cells:
        #if FR lable = cell type then use information of the current cell type
        if cell_type.name == g.attrs["label"]:
            current_cell_type = cell_type
            print("This is a ", cell_type.name)

    cell_num = network_scaffold.get_placed_count(current_cell_type.name)
    g_arr = np.array(g)
    print(g_arr) #array with N rows and 2 cols. cols = [spiking_cell_ID  FR_val]
    gc_ind = g_arr[:,1]>cutoff #check on FR: keep only ID with FR > cutoff frequency
    print(cutoff)
    gc = g_arr[gc_ind,:] #rimetto insieme FR e id che superano il cutoff
    tot_spikes = gc.shape[0]
    print("tot_spikes", tot_spikes, "for", cell_num, current_cell_type.name)

    unique, counts = np.unique(gc[:,0], return_counts=True)

    rate[current_cell_type.name]=counts/((duration-cutoff*0.001))

    print("population mean_rate", mean_rate)
    for c in rate.keys():
        print(c, "rate: ",np.mean(rate[c]),"+-",np.std(rate[c]))

        results = h5py.File(path+recordings_file, "r")
        print("results", results["/recorders/soma_spikes"])

    fig_psth = go.Figure()
    #duration is bin width
    ##---- quando uso SHORT meetto duration = 1
    print(cutoff)
    hdf5_plot_psth(network_scaffold, results["/recorders/soma_spikes"], duration=15, cutoff=cutoff, fig=fig_psth,
                   y_name=args.bsb_rec_file)

    # PLOT JUST PSTH -------------------------------------------------------------------------------------------
    if args.type_img == 1:
        title = 'PSTH'
        imtitle = 'PSTH_' + args.bsb_rec_file
        fig_psth.update_layout(
            title=title,
            autosize=True,
            #width=1000,
            #height=600,
            #margin=dict(l=5, r=50, b=1, t=100, pad=4),
            paper_bgcolor="white",
            font = dict(family="Arial", size=15),
        )

    #PLOT PSTH + MF ---------------------------------------------------------------------------------------------
    elif args.type_img == 2:
        mf_rec_file = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_prediction/'+args.bsb_rec_file.replace('.hdf5', '.npy')
        GrC_MF, GoC_MF, MLI_MF, PC_MF, _ = np.load(mf_rec_file, allow_pickle=True)

        title = 'PSTH_MF'
        imtitle = 'PSTH_MF_' + args.bsb_rec_file.replace('.hdf5', '')

        fig_psth.update_layout(
            title=title,
            autosize=True,
            # width=1000,
            # height=600,
            # margin=dict(l=5, r=50, b=1, t=100, pad=4),
            paper_bgcolor="white",
            font=dict(family="Arial", size=15),
        )


        #MLI on Baskets
        fig_psth.add_trace(go.Scatter(
                    x = np.arange(0, duration+1e-4, 1e-4)*1000,
                    y = MLI_MF,
                    mode="lines",
                    name = 'MLI_MF',
                    line=dict(color="salmon", width=2),
                    marker=dict(color="white", size=6, line=dict(color="black", width=8)),
                ), row=1, col=1)

        #GoC on GoC
        fig_psth.add_trace(go.Scatter(
                x = np.arange(0, duration+1e-4, 1e-4)*1000,
                y = GoC_MF,
                mode="lines",
                name = 'GoC_MF',
                line=dict(color="blue", width=2),
                marker=dict(color="white", size=6, line=dict(color="black", width=8)),
            ), row=2, col=1)

        #GrC on GrC
        fig_psth.add_trace(go.Scatter(
                x = np.arange(0, duration+1e-4, 1e-4)*1000,
                y = GrC_MF,
                mode="lines",
                name = 'GrC_MF',
                line=dict(color="red", width=2),
                marker=dict(color="white", size=6, line=dict(color="black", width=8)),
            ), row=3, col=1)

        #PC on PC
        fig_psth.add_trace(go.Scatter(
                x = np.arange(0, duration+1e-4, 1e-4)*1000,
                y = PC_MF,
                mode="lines",
                name = 'PC_MF',
                line=dict(color="green", width=2),
                marker=dict(color="white", size=6, line=dict(color="black", width=8)),
            ), row=4, col=1)

        #MLI on Stellate
        fig_psth.add_trace(go.Scatter(
                x = np.arange(0, duration+1e-4, 1e-4)*1000,
                y = MLI_MF,
                mode="lines",
                name = 'MLI_MF',
                line=dict(color="salmon", width=2),
                marker=dict(color="white", size=6, line=dict(color="black", width=8)),
            ), row=5, col=1)

        fig_psth.show()

    else: print("Type img not valid. Select 1 or 2")


    if args.save_imgs:
        fig_psth.write_image(os.getcwd()+'/imgs/'+imtitle+'.svg', scale=1, width=800, height=800)
        fig_psth.write_html(os.getcwd()+'/imgs/'+imtitle+'.html')#scale=1, width=400, height=400) #400 è mezzo A4. 800 per tutto. Se non va install ORCA
