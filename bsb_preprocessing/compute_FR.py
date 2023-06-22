import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bsb.core import from_hdf5
from collections import Counter
import os


def get_duration(network, sim_key):
    simulation = network.create_adapter(sim_key)
    return simulation.duration

def get_records(file_name, rec_cell):
    h = h5py.File(file_name, 'r')
    return h['recorders/soma_spikes/'+rec_cell]


def compute_spiking_cells(dataset):
    ids = list(dataset[:,0])
    d_count_id = Counter(ids) #{id: number of occurencies}
    list_count = np.array(list(d_count_id.values()))#to make an array from dict is required to pass from list
    return list_count


def comp_freq(dataset):
    """
    Output frequencies compute as number of spikes/duration of stimulation
    :param dataset: recorder of interest
    :return: f
    """
    list_count = compute_spiking_cells(dataset)
    duration = get_duration(network, sim_key)*1e-3 #duration is in ms! I need sec to have Hz
    print ('simulation lasts: ', duration)
    f = list_count/duration

    return f


def plt_freq_ddp(f, cell_type, input_freq):
    # Freedman Diaconis rule to determine bin width bw = 2*IQR(x)/cub_rad(n)
    n_bin = np.histogram_bin_edges(f, bins='fd')
    plt.hist(f, bins=n_bin)
    plt.title(cell_type+' - input[Hz]:  '+input_freq)
    #plt.show()


def compute_freq_loop(dir_results, cell_type_recorder, FOLDER, save_ddp=True):

    #Check where I am
    work_dir = os.getcwd()
    print("==================== Hi baby! Here I am: ",work_dir,"====================")

    # Check if the output folder already exists and if not create it:)
    if os.path.exists(work_dir + '/' +FOLDER):
        print('Folder %s has been already created!' % FOLDER)
    else:
        try:
            os.mkdir(FOLDER)
        except OSError:
            print("Creation of the directory %s failed" % FOLDER)
        else:
            print("Successfully created the directory %s " % FOLDER)


    # Change folder!!!! I go where results are saved
    os.chdir(dir_results)
    print("==================== Now I go into: ",os.getcwd(),"====================")

    file_list = os.listdir(os.getcwd())

    # loop on the results of bsb-nest sim
    for filename in file_list:

        if filename.startswith("wNET_results_stim_on_MFs_") & filename.endswith(".hdf5"):

            print('............... Working on: ',filename)

            #import output of simulations
            rec_cell = get_records(filename, cell_type_recorder)
            print('Check dimension: num cell spiking, spike timing: ', np.shape(rec_cell))

            #compute FR of EACH cells spiking
            f = comp_freq(rec_cell)
            print('Check frequencies dimension: ', len(f))

            #plot FR ddp
            if save_ddp:
                plt_freq_ddp(f, cell_type_recorder, filename)
                plt.savefig('/home/bcc/bsb-ws/CRBL_MF_Model/bsb_preprocessing/ddp_imgs/'+cell_type_recorder+filename+ '.png')
                plt.close()


            #compute statistics on FR
            #f_mean, f_sd, f_median, f_mode = compute_stats4freq(f)

            file_save_name = filename.replace('.hdf5', '_FR.npy')

            np.save(work_dir + '/' + FOLDER + '/' + file_save_name, f)
            print('===============================================================================================')

        else:
            print("======= No output hdf5 files from simulation have been found here! I'm super sorry baby ==========")
            continue

    os.chdir(work_dir)
    print('I am done and I am in :', os.getcwd())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Compute and saving Cell Firing Rate starting from bsb-nest output'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-DIR', help="DIR where results hdf5 are saved",
                        default='/home/bcc/bsb-ws/CRBL_MF_Model/bsb_preprocessing/res_MFs_0_80')

    parser.add_argument('-DIR_4_FR', help="DIR where to save FR results",
                        default='/home/bcc/bsb-ws/CRBL_MF_Model/bsb_preprocessing/DIR_4_FR')

    parser.add_argument('-net_hdf5', help="hdf file with net config",
                        default='balanced.hdf5')

    parser.add_argument('-sim_key', help="simulations of interest",
                        default='stim_on_MFs')

    parser.add_argument('-rec', help="cell recorders",
                        default='record_golgi_spikes')

    args = parser.parse_args()

    print('Cell recorders: ', args.rec)
    network = from_hdf5(args.net_hdf5)
    sim_key = args.sim_key


    def run_one_freq_procedure(hdf5file, input_freq):
        print('======================= Run Forest Run!!!! FR routine for one frequency input ======================= ')
        dataset = get_records(args.DIR+'/'+hdf5file, args.rec)
        print('Dataset considered in: '+args.DIR+'/'+hdf5file)
        FR_vec = comp_freq(dataset)
        print('Mean +- sd: ', np.mean(FR_vec), ' +- ', np.std(FR_vec),'\nMedian', np.median(FR_vec))
        plt_freq_ddp(FR_vec, args.rec, input_freq)
        return FR_vec


    def check_npy_file(cellfolder_npyfile):
        FR_vec = np.load(os.getcwd()+'/'+cellfolder_npyfile,
                     allow_pickle=True)
        print('Mean Freq: ', FR_vec.mean())
        plt_freq_ddp(FR_vec, '', cellfolder_npyfile)
        plt.show()


    #check_npy_file('STELL_FR/wNET_results_stim_on_MFs_input12_FR.npy')
    #FR_vec = run_one_freq_procedure(hdf5file = 'wNET_results_stim_on_MFs_input20mfs52.hdf5', input_freq = str(52))


    ## ---------- Compute population Firing Rate (FR) for each Mossy fibers input.
    compute_freq_loop(args.DIR, args.rec, args.DIR_4_FR)