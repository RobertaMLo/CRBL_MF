from bsb.config import JSONConfig
from bsb.output import HDF5Formatter
import h5py
from bsb.core import from_hdf5
import os

network_file = "balanced4VALID.hdf5"
nest_config = JSONConfig("mouse_cerebellum_cortex_update_copy_post_stepwise_colonna_X_4VALIDATION.json")

HDF5Formatter.reconfigure(network_file, nest_config)
network_scaffold = from_hdf5(network_file)

simulation = network_scaffold.create_adapter("stim_on_MFs")

print("============================================")
print("Configuration: ", network_scaffold)
print('Sim lasts[ms]: ', simulation.duration)
print("============================================")

#print(len(simulation.devices))
print("============================================"
      "\nSTIMULATION PROTOCOL")
print('Background Noise:\nRate [Hz] = ',simulation.devices["background_noise"].parameters["rate"],
      '\nt0 [ms] = ',simulation.devices["background_noise"].parameters["start"],
      '\nt_end [ms] = ',simulation.devices["background_noise"].parameters["stop"])

print('Spikes from mf:\nRate [Hz] = ',simulation.devices["tone_stim"].parameters["rate"],
      '\nt0 [ms] = ',simulation.devices["tone_stim"].parameters["start"],
      '\nt_end [ms] = ',simulation.devices["tone_stim"].parameters["stop"])


simulator = simulation.prepare()
simulation.simulate(simulator)
data_file = simulation.collect_output(simulator)

my_new_name = "wNET_4VALID_results.hdf5"
os.rename(os.getcwd()+"/"+data_file, os.getcwd()+"/"+my_new_name)

with h5py.File(my_new_name, "a") as f:
    print("Captured", len(f["recorders"]))