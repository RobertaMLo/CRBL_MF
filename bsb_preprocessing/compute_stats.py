import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from statistics import mean, stdev, median, mode


def compute_stats4freq(f):
    f_mean, f_sd = mean(f), stdev(f)
    f_median = median(f)
    f_mode = mode(f)
    print('FR [Hz] statistics')
    print('Mean +- sd = '+str(f_mean)+' +- '+str(f_sd))
    print('Median = '+str(f_median))
    print('Mode = '+str(f_mode))

    return f_mean, f_sd, f_median, f_mode


def compute_range(f_mean, f_sd, nbin = 10):
    first_half = np.arange(f_sd, f_mean, int(nbin/2))
    second_half = np.arange(f_mean, f_sd, int(nbin/2))
    range_f = np.array([first_half, second_half])
    return range_f


def loop_on_fr(DIR_FR):
    os.chdir(DIR_FR)
    list_file = os.listdir(os.getcwd())

    #load FR for each stimulation
    n_stim = int((MF_last - MF_first)/step_stim)+1

    f_mean_vec = np.zeros([n_stim])
    f_sd_vec = np.zeros([n_stim])
    f_median_vec = np.zeros([n_stim])
    f_mode_vec = np.zeros([n_stim])

    input_MF = np.arange(MF_first, MF_last+1, step_stim)

    # loop on input range
    for i in range(len(input_MF)):

        str_to_find = "_input"+str(int(input_MF[i]))+"_FR.npy"
        #str_to_find = "_K35_" + str(int(input_MF[i])) + "_FR.npy"
        print('looking for ',str_to_find)

        # loop on files
        for j in range(len(list_file)):
            if str_to_find in list_file[j]:

                print('Working on ', list_file[j])
                f = np.load(list_file[j], allow_pickle=True)

                f_mean, f_sd, f_median, f_mode = compute_stats4freq(f)

                f_mean_vec[i], f_sd_vec[i], f_median_vec[i], f_mode_vec[i] = f_mean, f_sd, f_median, f_mode

        print('=====================================================================================')

    os.chdir(work_dir)

    return f_mean_vec, f_sd_vec, f_median_vec, f_mode_vec


def write_csv_4_stats(f_mean_vec, f_sd_vec, f_median_vec, f_mode_vec, folder, filename_csv):
    import csv
    path = folder+'/'+filename_csv
    #path = '/home/bcc/bsb-ws/CRBL_MF_Model/'+folder+'/'+filename_csv

    MF_inputs = np.arange(MF_first,MF_last+step_stim,step_stim)
    row = np.zeros([len(MF_inputs), 5])

    for i in range(len(MF_inputs)):

        row[i, :] = [MF_inputs[i], f_mean_vec[i], f_sd_vec[i], f_median_vec[i], f_mode_vec[i]]
        print(row[i,:])

    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(row)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Stats on FR compute from BSB sim'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-DIR_FR', help="DIR where FRs are saved",
                        default='default_input_dir')

    parser.add_argument('-MFs_first', help="first input to consider", type=float,
                        default=0.)
    parser.add_argument('-MFs_last', help="last input to consider", type=float,
                        default=80.)
    parser.add_argument('-step_stim', help="step between stim", type=float,
                        default=4.)

    parser.add_argument('-DIR_OUT', help="Directory where to save the results",
                        default='default_output_dir')
    parser.add_argument('-f_name', help="csv filename",
                        default='my_csv_file.csv')

    args = parser.parse_args()
    # Load the network_scaffold -- need for compute FR functions


    work_dir = os.getcwd()

    DIR_FR = work_dir +'/'+args.DIR_FR
    MF_first = args.MFs_first
    MF_last = args.MFs_last
    step_stim = args.step_stim

    print('===========================================================================================================\n'
          'Firing Population: ',DIR_FR,
          '\n==========================================================================================================\n')

    ## ------------------------ Compute the FR for each mossy input.
    fmean, fsd, fmedian, fmode = loop_on_fr(DIR_FR)

    #plt.figure(1), plt.plot(fmean), plt.plot(fmedian), plt.plot(fmode), plt.show()

    ## ------------------------ Writing a .CSV file to save statistics
    new_FOLDER, filename_csv = args.DIR_OUT, args.f_name

    if os.path.exists(work_dir + '/' +new_FOLDER):
        print('Folder %s has been already created!' % new_FOLDER)
    else:
        try:
            os.mkdir(new_FOLDER)
        except OSError:
            print("Creation of the directory %s failed" % new_FOLDER)
        else:
            print("Successfully created the directory %s " % new_FOLDER)

    write_csv_4_stats(fmean, fsd, fmedian, fmode, new_FOLDER, filename_csv)
