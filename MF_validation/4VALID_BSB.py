import h5py
from bsb.core import from_hdf5
import numpy as np

import os
from bsb.output import HDF5Formatter
from bsb.config import JSONConfig
import time
tstart_sim  =time.time()

#hdf5_abs_path = "/home/bcc/bsb-ws/CRBL_MF_Model/balanced.hdf5"
#hdf5_abs_path = "/home/bcc/bsb-ws/CRBL_MF_Model/balanced_reduced_mfgoc.hdf5"
hdf5_abs_path = "/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/balanced.hdf5"

#config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_UPDOWN3.json'
#config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_config_healthy.json'

#config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_UPDOWN1_500.json'

#config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_TRYSYN.json'
config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_SIN_500.json'
#recordings_file = '4paper_updown_trysyn_500.hdf5'
#recordings_file = '4paper_trysyn_500.hdf5'
recordings_file = '4paper_syn_500aaa.hdf5'

#config = '/home/bcc/bsb-ws/CRBL_MF_Model/MF_validation/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_SIN_20_desync.json'
#recordings_file = 'SIN_20_desync_wNET_results_40.hdf5'


reconfigured_obj = JSONConfig(config)
HDF5Formatter.reconfigure(hdf5_abs_path, reconfigured_obj)
print("reconfig done")


network = from_hdf5(hdf5_abs_path)
simulation = network.create_adapter("stim_on_MFs")

print('Sim lasts[s]: ', simulation.duration*1e-3)

simulator = simulation.prepare()
simulation.simulate(simulator)
data_file = simulation.collect_output(simulator)



os.rename(os.getcwd()+"/"+data_file, os.getcwd()+"/"+recordings_file)

with h5py.File(recordings_file, "a") as f:
        print("Captured", len(f["recorders"]))


time_sim = tstart_sim - time.time()
print('simulation last: ',time_sim)