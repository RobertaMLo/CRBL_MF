"""
Coupled Mean-Field Network for cerebellar cortex (first-order formalism).
External input is SUMMED to the transfer function input of the same nature:
TF(fe, fi) --> TF (fe + fe_ext , fi)

 - Each "node" is a cerebellar MF model (GrC / GoC / MLI / PC) defined as in Lorenzi et al., 2023 Plos Comp Biol.
 - Nodes are connected via a coupling matrix W_net such that the pure_exc_aff received by node j at time t is:
                        pure_exc_aff_j(t) = sum_i  [W_net[i, j] * X_i(t-dt, pop)]

X_i(t-dt, pop) is the pop firing rate of node i at the previous time step.
In the case of cerebellar mfm pop = GrC: Intra-cortical coupling are only excitatory and via GrC parallel fibers.
NB.: here's no parallel fibers granularity, so is GrC axons!!!!

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from master_equation_CRBL_MF_2 import build_up_differential_operator_first_order
from MF_prediction.input_library import *

# ---------------------------------------------------------------------------
# Core network solver - as for single MFM node
# ---------------------------------------------------------------------------

def find_fixed_point_network(nodes, W_net, t, w, T,
                             inh_aff_list=None, verbose=True):
    """
    Simulate a network of N coupled first-order cerebellar MF modules.

    Parameters
    ----------
    nodes : list of dict, length N
        Idea is that one node is a MFM, and it is described by the fitted TF and the CI.
        *** TO BE TESTED: if nodes are different MFM it should work as well ***
        Also exc-aff is added because it can differ nodes by nodes

        Mandatory keys:
        'TF': transfer functions (callable) - loaded from load_config_TF
        'CI': initial conditions, array-like (length 4 for cerebellar cortex)
        'exc_aff' : array of length len(t), mossy-fiber drive for this node

        Optional keys:
            'inh_aff' : array of length len(t), inhibitory afferent (default 0) - for flexibility
            'coupling_weight' : scalar weight applied to incoming signal (default 1.0, useful for per-node gain)


    W_net : np.ndarray, shape (N, N)
        Connectivity matrix.  W_net[i, j] is the weight of the projection
        from node i to node j (GrC of node i → pure_exc_aff of node j).
        Diagonal entries are ignored - no self-coupling.

    t : np.ndarray
        Time vector.

    w : float
        Adaptation parameter passed to all TFs. Default is 0, but just in case we want to add adap to crbl MFM...

    T : float
        Membrane time constant (same for all nodes, derived as in Lorenzi et al., 2023).

    inh_aff_list : list of arrays or None
        Per-node inhibitory afferent arrays of length len(t). If None, all inhibitory afferents are 0.
        Here it is None because intra-cerebellar cortical connections are only excitatory

    verbose : bool
        Print final state of each node.

    Returns
    -------
    X_all : list of np.ndarray, length N
        X_all[n] has shape (len(t), 4):
            col 0 = GrC,  col 1 = GoC,  col 2 = MLI,  col 3 = PC
    """

    N = len(nodes)
    dt = t[1] - t[0]

    # ----- Differential operators for each node -----------------------------
    diff_ops = []

    for node in nodes:
        op = build_up_differential_operator_first_order(
            node['TF1'], node['TF2'], node['TF3'], node['TF4'], w, T
        )
        diff_ops.append(op)

    # ----- State variables vectors -------------------------------------------
    # All pops
    X_all = [np.zeros((len(t), 4)) for _ in range(N)]
    # Current pop
    X_cur = [np.array(node['CI'], dtype=float) for node in nodes]
    print('SHAPE STATE VAR VECTORS: X all', np.shape(X_all), 'X_cur', np.shape(X_cur))

    # Write initial conditions
    for n in range(N):
        X_all[n][0, :] = X_cur[n]

    # ----- Afferents Connection -----------------------------------------------
    #Inhibithory
    # if in_aff is not None must be implemented!!!!
    if inh_aff_list is None:
        inh_aff_list = [np.zeros(len(t)) for _ in range(N)]

    # Excitatory input to the connected nodes
    exc_aff_list = [np.asarray(node['exc_aff']) for node in nodes]

    # ----- Let's solve the mean field model --------------------------------
    for i in range(1, len(t)):

        # THIS IS THE NEW PART !!!!!!
        # Compute pure_exc_aff for every node n using GrC activity at t[i-1]. The coupling is one-step delayed!!!!
        # grc_prev[n] = X_all[n][i-1, 0]
        grc_prev = np.array([X_all[n][i - 1, 0] for n in range(N)])
        #print('max of grc_prev', np.max(grc_prev))

        # pure_exc_aff_j = sum_i  W_net[i, j] * grc_prev[i]
        # matrix product as "rows*cols" implemented with @
        pure_exc_aff = W_net.T @ grc_prev   # shape (N,)
        #print('shape input', np.shape(pure_exc_aff))

        # Integrate each node
        for n in range(N):
            coupling_weight = nodes[n].get('coupling_weight', 1.0)
            dX = diff_ops[n](
                X_cur[n],
                exc_aff=exc_aff_list[n][i], # this is fmossy
                inh_aff=inh_aff_list[n][i], # this is not present in cerebellums
                pure_exc_aff=coupling_weight * pure_exc_aff[n] #this is grc of another node
            )

            # INTEGRATOR: SIMPLE EULER
            X_cur[n] = X_cur[n] + dt * dX
            X_all[n][i, :] = X_cur[n]

    if verbose:
        for n in range(N):
            print(f"Node {n} final state  "
                  f"GrC={X_all[n][-1, 0]:.3f}  GoC={X_all[n][-1, 1]:.3f}  "
                  f"MLI={X_all[n][-1, 2]:.3f}  PC={X_all[n][-1, 3]:.3f}")

    return X_all


def make_W_unidirectional(N, source, target, weight=1.0):
    """
    Single directed connection: source → target.
    Might be useful considering cureted connections in the cerebellum. E.g., PC to DCN but NO DCN to PC.

    Parameters
    ----------
    N      : total number of nodes
    source : index of the sending node
    target : index of the receiving node
    weight : synaptic weight

    Returns
    -------
    W : np.ndarray (N, N)
    """
    W = np.zeros((N, N))
    W[source, target] = weight
    return W


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from MF_prediction.load_config_TF import load_transfer_functions, load_transfer_functions_goc
    from MF_prediction.input_library import rect_input

    # ---- Paths & TF loading (adjust to your environment) ------------------
    root_path = os.getcwd() +'/'
    print(root_path)
    NTWK = 'CRBL_CONFIG_20PARALLEL_wN'

    FILE_GrC = root_path + '20220519_120033_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    FILE_GoC = root_path + '20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
    FILE_MLI = root_path + '20220622_085550_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.8_fit.npy'
    FILE_PC  = root_path + '20220622_085610_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.5_fit.npy'

    TFgrc = load_transfer_functions('GrC', NTWK, FILE_GrC, alpha=2.0)
    TFgoc = load_transfer_functions_goc('GoC', NTWK, FILE_GoC, alpha=1.3)
    TFmli = load_transfer_functions('MLI', NTWK, FILE_MLI, alpha=5)
    TFpc  = load_transfer_functions('PC',  NTWK, FILE_PC,  alpha=5)

    # ---- Simulation parameters --------------------------------------------
    w  = 0
    T  = 3.5e-3
    dt = 1e-4
    t  = np.arange(0, 0.5, dt)

    # Two different mossy-fiber drives
    f_tone1 = rect_input(time=t, t_start=1000, t_end=1500, minval=0, freq=100, noise_freq=0)
    f_tone2 = rect_input(time=t, t_start=2000, t_end=2500, minval=0, freq=100, noise_freq=0)
    f_tone3 = rect_input(time=t, t_start=3000, t_end=3500, minval=0, freq=100, noise_freq=0)
    fmossy_A = np.random.rand(len(t)) * 4 + f_tone1 + f_tone2 + f_tone3

    #fmossy_A = np.random.rand(len(t)) * 100   # node A: 20 Hz background
    fmossy_B = np.random.rand(len(t)) * 4   # node B:  5 Hz background

    fmossy_C = np.ones_like(fmossy_A)*fmossy_A*0.5  # node B:  5 Hz background

    # Initial condition set as in Lorenzi et al., 2023, Plos Comp Biol.
    CI = [0.5, 5, 15, 38]   # [GrC_0, GoC_0, MLI_0, PC_0]

    # ---- Connectivity ----------------------
    # W_net[i, j] = weight of projection from node i to node j
    # mock example: B = vermis, receiving from hemisphere R and L (A and C)

    N_nodes = 3
    W_net = make_W_unidirectional(N_nodes, source=0, target=1, weight=1.0) + make_W_unidirectional(N_nodes, source=2, target=1, weight=1.0)

    print("Coupling matrix W_net:\n", W_net)

    # ---- Run network simulation -------------------------------------------
    nodes = [
        {'TF1': TFgrc, 'TF2': TFgoc, 'TF3': TFmli, 'TF4': TFpc,
         'CI': CI, 'exc_aff': fmossy_A},
        {'TF1': TFgrc, 'TF2': TFgoc, 'TF3': TFmli, 'TF4': TFpc,
         'CI': CI, 'exc_aff': fmossy_B},
        {'TF1': TFgrc, 'TF2': TFgoc, 'TF3': TFmli, 'TF4': TFpc,
         'CI': CI, 'exc_aff': fmossy_C}
    ]

    X_all = find_fixed_point_network(nodes, W_net, t, w, T, verbose=True)


    # ---- Quick plot -------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ['GrC', 'GoC', 'MLI', 'PC']
    colors = ['red', 'blue', 'orange', 'green']

    for k in range(4):
        axes[k].plot(t, X_all[0][:, k], color=colors[k], linestyle='--', label=f'Node A')
        axes[k].plot(t, X_all[1][:, k], color=colors[k], label=f'Node B')
        axes[k].plot(t, X_all[2][:, k], color=colors[k], linestyle=':', label=f'Node C')

        axes[k].set_ylabel(f'$\\nu_{{{labels[k]}}}$ [Hz]', fontsize=18)
        axes[k].tick_params(axis='y', labelsize=15)

        axes[k].legend(fontsize=15, loc='upper right')

    axes[-1].tick_params(axis='x', labelsize=15)
    axes[-1].set_xlabel('time [s]', fontsize=18)
    fig.suptitle('CRBL MF Network  –  Node A → Node B', fontsize=20)
    plt.tight_layout()
    plt.show()