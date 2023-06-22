import h5py
from bsb.core import from_hdf5
import numpy as np
import os
from bsb.config import JSONConfig
from bsb.output import HDF5Formatter
import h5py
from bsb.core import from_hdf5
import os

"""
#Vecchio codice
network = from_hdf5("balanced_reduced_mfgoc.hdf5")
simulation = network.create_adapter("stim_on_MFs")
print('Sim lasts[s]: ', simulation.duration*1e-3)
"""

#RECONFIG CON JSON

network_file = "/home/bcc/bsb-ws/CRBL_MF_Model/balanced_reduced_mfgoc_Ali05.hdf5"
json_file = "/home/bcc/bsb-ws/CRBL_MF_Model/mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X _Ali05.json"
network_scaffold = from_hdf5(network_file)
nest_config = JSONConfig(json_file)
HDF5Formatter.reconfigure(network_file, nest_config)
print("reconfig done")
simulation = network_scaffold.create_adapter("stim_on_MFs")


#VARIE PROVE SENZA RECONFIG
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_old2.hdf5")
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_reduced_mfgoc.hdf5")
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_old_1805_3.hdf5") #1.6
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_reduced_mfgoc_1805_2.hdf5") #1.6
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_old_1805_3.hdf5") #1.6
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_cla_05.hdf5") #5 ma err su stim_on_MFs
#network = from_hdf5("/home/bcc/bsb-ws/CRBL_MF_Model/balanced_06.hdf5") #5 ma err su glomeruli not founded
#simulation = network.create_adapter("stim_on_MFs")


print('Sim lasts[s]: ', simulation.duration*1e-3)
for i in np.arange(0,81,4):
    simulation.reset()
    simulation.devices["background_noise"].parameters["rate"] = i * 1.0 #float required
    print('============================== '
          'stimulation input = ',simulation.devices["background_noise"].parameters["rate"])

    simulator = simulation.prepare()
    simulation.simulate(simulator)
    data_file = simulation.collect_output(simulator)

    my_new_name = "wNET_results_stim_on_MFs_input"+str(i)+".hdf5"
    os.rename(os.getcwd()+"/"+data_file, os.getcwd()+"/"+my_new_name)

    with h5py.File(my_new_name, "a") as f:
        print("Captured", len(f["recorders"]))