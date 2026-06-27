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
from params_library.syn_and_connec_library import *
from master_equation_CRBL_MF_2 import build_up_differential_operator_first_order
from MF_prediction.input_library import *

# ---------------------------------------------------------------------------
# Core network solver - as for single MFM node
# ---------------------------------------------------------------------------

"""
network_CRBL_MF.py
------------------
Coupled Mean-Field Network for cerebellar cortex (first-order formalism).

Each "node" is one instance of the cerebellar MF model (GrC / GoC / MLI / PC).
Nodes are connected via a coupling matrix W_net such that the pure_exc_aff
received by node j at time t is:

    pure_exc_aff_j(t) = sum_i  W_net[i, j] * X_i(t-dt, 0)

where X_i(t-dt, 0) is the GrC firing rate of node i at the previous time step.

The coupling is therefore one-step delayed (Euler integration), which is the
natural and numerically consistent choice for explicit Euler schemes.

Usage example (2 nodes, unidirectional A→B):
----------------------------------------------
    W_net = np.array([[0.0, 1.0],   # node A sends to node B
                       [0.0, 0.0]]) # node B does not send back

    result = find_fixed_point_network(
        nodes=[
            {'TF1': TFgrc, 'TF2': TFgoc, 'TF3': TFmli, 'TF4': TFpc,
             'CI': [0.5, 10, 8.5, 10], 'exc_aff': fmossy_A},
            {'TF1': TFgrc, 'TF2': TFgoc, 'TF3': TFmli, 'TF4': TFpc,
             'CI': [0.5, 10, 8.5, 10], 'exc_aff': fmossy_B},
        ],
        W_net=W_net,
        t=t, w=w, T=T
    )
    # result is a list: result[node_idx] -> array of shape (len(t), 4)
"""

# ---------------------------------------------------------------------------
# Core network solver
# ---------------------------------------------------------------------------

def find_fixed_point_network(nodes, W_net, t, w, T,
                             inh_aff_list=None, verbose=True):
    """
    Simulate a network of N coupled first-order cerebellar MF modules.

    Parameters
    ----------
    nodes : list of dict, length N
        Each dict describes one node and must contain:
            'TF1', 'TF2', 'TF3', 'TF4' : transfer functions (callable)
            'CI'      : initial conditions, array-like of length 4
                        [GrC_0, GoC_0, MLI_0, PC_0]
            'exc_aff' : array of length len(t), mossy-fiber drive for this node
        Optional keys:
            'inh_aff' : array of length len(t), inhibitory afferent (default 0)
            'coupling_weight' : scalar weight applied to *incoming* signal
                                BEFORE W_net (default 1.0, useful for per-node
                                gain without changing W_net)

    W_net : np.ndarray, shape (N, N)
        Connectivity matrix.  W_net[i, j] is the weight of the projection
        from node i to node j (GrC of node i → pure_exc_aff of node j).
        Diagonal entries are ignored (no self-coupling via this mechanism).

    t : np.ndarray
        Time vector.

    w : float
        Adaptation/noise parameter passed to all TFs.

    T : float
        Membrane time constant (same for all nodes, can be extended).

    inh_aff_list : list of arrays or None
        Per-node inhibitory afferent arrays of length len(t).
        If None, all inhibitory afferents are 0.

    verbose : bool
        Print final state of each node.

    Returns
    -------
    X_all : list of np.ndarray, length N
        X_all[n] has shape (len(t), 4):
            col 0 = GrC,  col 1 = GoC,  col 2 = MLI,  col 3 = PC
    """

    N = len(nodes)
    assert W_net.shape == (N, N), "W_net must be (N, N)"

    dt = t[1] - t[0]

    # ----- Build differential operators for each node ----------------------
    diff_ops = []
    for node in nodes:
        op = build_up_differential_operator_first_order(
            node['TF1'], node['TF2'], node['TF3'], node['TF4'], w, T
        )
        diff_ops.append(op)

    # ----- Initialise state vectors ----------------------------------------
    X_all = [np.zeros((len(t), 4)) for _ in range(N)]
    X_cur = [np.array(node['CI'], dtype=float) for node in nodes]

    # Write initial conditions
    for n in range(N):
        X_all[n][0, :] = X_cur[n]

    # ----- Default afferents -----------------------------------------------
    if inh_aff_list is None:
        inh_aff_list = [np.zeros(len(t)) for _ in range(N)]

    exc_aff_list = [np.asarray(node['exc_aff']) for node in nodes]

    # ----- Time integration (explicit Euler) --------------------------------
    for i in range(1, len(t)):

        # Compute pure_exc_aff for every node using GrC activity at t[i-1]
        # grc_prev[n] = X_all[n][i-1, 0]
        grc_prev = np.array([X_all[n][i - 1, 0] for n in range(N)])

        # pure_exc_aff_j = sum_i  W_net[i, j] * grc_prev[i]
        # (column-wise dot product)
        pure_exc_aff = W_net.T @ grc_prev   # shape (N,)

        # Integrate each node
        for n in range(N):
            coupling_weight = nodes[n].get('coupling_weight', 1.0)
            dX = diff_ops[n](
                X_cur[n],
                exc_aff=exc_aff_list[n][i],
                inh_aff=inh_aff_list[n][i],
                pure_exc_aff=coupling_weight * pure_exc_aff[n]
            )
            X_cur[n] = X_cur[n] + dt * dX
            X_all[n][i, :] = X_cur[n]

    if verbose:
        for n in range(N):
            print(f"Node {n} final state  "
                  f"GrC={X_all[n][-1, 0]:.3f}  GoC={X_all[n][-1, 1]:.3f}  "
                  f"MLI={X_all[n][-1, 2]:.3f}  PC={X_all[n][-1, 3]:.3f}")

    return X_all


# ---------------------------------------------------------------------------
# Convenience: build standard coupling matrices
# ---------------------------------------------------------------------------

def make_W_unidirectional(N, source, target, weight=1.0):
    """
    Single directed connection: source → target.

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


def make_W_chain(N, weight=1.0, recurrent=False):
    """
    Linear chain:  0 → 1 → 2 → ... → N-1
    If recurrent=True, also adds N-1 → 0.
    """
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = weight
    if recurrent:
        W[N - 1, 0] = weight
    return W


def make_W_all_to_all(N, weight=1.0, include_self=False):
    """
    All-to-all connectivity.
    """
    W = np.full((N, N), weight)
    if not include_self:
        np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# Cortex <-> DCN extension (heterogeneous nodes: 4 pops vs 2 pops)
# ---------------------------------------------------------------------------
#
# The cortex node (GrC, GoC, MLI, PC) and the DCN node (DCNp, DCNi) have
# DIFFERENT numbers of state variables, so they cannot share the same
# square W_net used for same-sized nodes above. We use a dedicated function
# instead of forcing them into the generic N-node framework.
#
# Circuit assumptions (see conversation):
#   - fmossy_cortex  -> exc_aff of cortex node (as in the original model)
#   - fmossy_DCN     -> exc_aff of DCNp (direct mossy fiber collateral)
#   - PC(t-1)        -> inh_aff of DCNp  (weight w_PC_DCNp)
#   - PC(t-1)        -> inh_aff of DCNi  (weight w_PC_DCNi)
#   - DCNi has NO explicit excitatory afferent (fe=0 passed to TF_DCNi)
#   - one-way coupling: cortex -> DCN only, no feedback DCN -> cortex
# ---------------------------------------------------------------------------
def build_up_differential_operator_DCN(TF_DCNp, TF_DCNi, w, T):
    """
    First-order differential operator for the DCN node.

    State vector V = [DCNp, DCNi]   (V[0], V[1])

    TF_DCNp(fe_mf, fe_grc, fi, w) : fe_mf = mossy collateral (exc_aff_mf),
                                     fe_grc = granule cell afferent from cortex (exc_aff_grc),
                                     fi = PC inhibition
    TF_DCNi(fe, fi, w)            : fe is forced to 0 (no excitatory afferent),
                                     fi = PC inhibition
    """

    def Ap(V, exc_aff_mf, exc_aff_grc, inh_aff):
        return (1. / T) * (TF_DCNp(exc_aff_mf, exc_aff_grc, inh_aff, w) - V[0])

    def Ai(V, inh_aff):
        # no excitatory afferent -> fe = 0
        return (1. / T) * (TF_DCNi(inh_aff, 0) - V[1])

    def Diff_OP(V, exc_aff_mf, exc_aff_grc, inh_aff_p, inh_aff_i):
        return np.array([Ap(V, exc_aff_mf=exc_aff_mf, exc_aff_grc=exc_aff_grc, inh_aff=inh_aff_p),
                         Ai(V, inh_aff=inh_aff_i)])

    return Diff_OP


def find_fixed_point_DCN(TF_DCNp, TF_DCNi,
                         CI_DCN, t, w, T,
                         fe_mossy, fe_grc, fi,
                         verbose=True):
    """
    Simulate the DCN mean-field node IN ISOLATION (no cerebellar cortex),
    with the afferent firing rates set directly by the user.

    State vector V = [DCNp, DCNi], integrated as in
    build_up_differential_operator_DCN (first-order, explicit Euler):

        TF_DCNp(fe_mf, fe_grc, fi, w)  -> DCNp
        TF_DCNi(0,     fi,     w)      -> DCNi   (fe forced to 0, as before)

    Parameters
    ----------
    TF_DCNp, TF_DCNi : transfer functions for DCNp and DCNi
    CI_DCN : array-like, len 2   -> [DCNp_0, DCNi_0]
    t : time vector
    w : adaptation/noise parameter
    T : membrane time constant
    fe_mossy : scalar or array, len(t)
        Mossy-fiber collateral afferent to DCNp (exc_aff_mf).
    fe_grc : scalar or array, len(t)
        GrC-like excitatory afferent to DCNp (exc_aff_grc), fed in as the
        raw firing rate the user wants to test (then scaled by
        w_GrC_DCNp, default 1.0, i.e. used as-is).
    fi : scalar or array, len(t)
        Inhibitory (PC-like) afferent, applied to BOTH DCNp and DCNi
        (each scaled by its own weight, w_PC_DCNp / w_PC_DCNi, default
        1.0, i.e. used as-is for both).
    w_PC_DCNp, w_PC_DCNi : scalar weights applied to `fi` for DCNp / DCNi
    w_GrC_DCNp : scalar weight applied to `fe_grc` for DCNp
    verbose : bool
        Print the final (steady-state, if t is long enough) DCNp/DCNi rates.

    Returns
    -------
    X_DCN : np.ndarray, shape (len(t), 2)
        col 0 = DCNp,  col 1 = DCNi
    """

    dt = t[1] - t[0]
    n = len(t)

    def _as_array(x, name):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full(n, float(x))
        if x.shape[0] != n:
            raise ValueError(
                f"{name} must be a scalar or an array of length len(t) ({n}), got shape {x.shape}"
            )
        return x

    fe_mossy_arr = _as_array(fe_mossy, 'fe_mossy')
    fe_grc_arr = _as_array(fe_grc, 'fe_grc')
    fi_arr = _as_array(fi, 'fi')

    diff_op_DCN = build_up_differential_operator_DCN(TF_DCNp, TF_DCNi, w, T)

    X_DCN = np.zeros((n, 2))
    Xd = np.array(CI_DCN, dtype=float)
    X_DCN[0, :] = Xd

    for i in range(1, n):
        dXd = diff_op_DCN(
            Xd,
            exc_aff_mf=fe_mossy_arr[i],
            exc_aff_grc=fe_grc_arr[i],
            inh_aff_p=fi_arr[i],
            inh_aff_i=fi_arr[i]
        )
        Xd = Xd + dt * dXd
        X_DCN[i, :] = Xd

    if verbose:
        print(f'DCN-only final:  DCNp={X_DCN[-1, 0]:.3f}  DCNi={X_DCN[-1, 1]:.3f}')

    return X_DCN


def find_fixed_point_cortex_DCN_network(TFgrc, TFgoc, TFmli, TFpc,
                                        TF_DCNp, TF_DCNi,
                                        CI_cortex, CI_DCN,
                                        t, w, T,
                                        fmossy_cortex, fmossy_DCN,
                                        w_PC_DCNp=1.0,
                                        w_PC_DCNi=1.0,
                                        w_GrC_DCNp=1.0,
                                        verbose=True):
    """
    Simulate the coupled cortex + DCN network (first-order, explicit Euler).

    Parameters
    ----------
    TFgrc, TFgoc, TFmli, TFpc : transfer functions for the cortex node
    TF_DCNp, TF_DCNi          : transfer functions for the DCN node
    CI_cortex : array-like, len 4   -> [GrC_0, GoC_0, MLI_0, PC_0]
    CI_DCN    : array-like, len 2   -> [DCNp_0, DCNi_0]
    t  : time vector
    w  : adaptation/noise parameter (shared)
    T  : membrane time constant (shared)
    fmossy_cortex : array, len(t)  -> mossy drive to cortex (GrC/GoC), as before
    fmossy_DCN    : array, len(t)  -> mossy collateral drive to DCNp directly
    w_PC_DCNp, w_PC_DCNi : scalar weights for the PC -> DCNp / DCNi inhibition
    w_GrC_DCNp            : scalar weight for the GrC -> DCNp excitation

    Returns
    -------
    X_cortex : np.ndarray, shape (len(t), 4)   columns = [GrC, GoC, MLI, PC]
    X_DCN    : np.ndarray, shape (len(t), 2)   columns = [DCNp, DCNi]
    """

    dt = t[1] - t[0]

    diff_op_cortex = build_up_differential_operator_first_order(TFgrc, TFgoc, TFmli, TFpc, w, T)
    diff_op_DCN = build_up_differential_operator_DCN(TF_DCNp, TF_DCNi, w, T)

    X_cortex = np.zeros((len(t), 4))
    X_DCN = np.zeros((len(t), 2))

    Xc = np.array(CI_cortex, dtype=float)
    Xd = np.array(CI_DCN, dtype=float)

    X_cortex[0, :] = Xc
    X_DCN[0, :] = Xd

    for i in range(1, len(t)):

        PC_prev = X_cortex[i - 1, 3]    # PC activity at t-1 (one-step delayed coupling)
        GrC_prev = X_cortex[i - 1, 0]   # GrC activity at t-1 (one-step delayed coupling)

        # ----- cortex update (unchanged, no feedback from DCN) -----
        dXc = diff_op_cortex(Xc, exc_aff=fmossy_cortex[i])
        Xc = Xc + dt * dXc
        X_cortex[i, :] = Xc

        # ----- DCN update -----
        dXd = diff_op_DCN(
            Xd,
            exc_aff_mf=fmossy_DCN[i],
            exc_aff_grc=w_GrC_DCNp * GrC_prev,
            inh_aff_p=w_PC_DCNp * PC_prev,
            inh_aff_i=w_PC_DCNi * PC_prev
        )

        Xd = Xd + dt * dXd
        X_DCN[i, :] = Xd

    if verbose:
        print(f'Cortex final:  GrC={X_cortex[-1,0]:.3f}  GoC={X_cortex[-1,1]:.3f}  '
              f'MLI={X_cortex[-1,2]:.3f}  PC={X_cortex[-1,3]:.3f}')
        print(f'DCN final:     DCNp={X_DCN[-1,0]:.3f}  DCNi={X_DCN[-1,1]:.3f}')

    return X_cortex, X_DCN

# # ----------------------------------------------------------------------
# # PLOTTING PART
# # ----------------------------------------------------------------------
POP_COLORS = {
    'input': 'black',
    'GrC':   'red',
    'GoC':   'blue',
    'MLI':   'orange',
    'PC':    'green',
    'DCNi':  'hotpink',
    'DCNp':  'purple',
}

def _as_node_list(x):
    """Normalize a single array or a list/tuple of arrays into a list of arrays
    (one entry per cortex node)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"Unsupported type for node data: {type(x)}")


def plot_coupled_mfm_network(t, fmossy_cortex, X_cortex,
                              fmossy_DCN=None, X_DCN=None,
                              node_labels=None, time_unit='ms',
                              figsize=None, suptitle=None,
                              save_path=None, show=True):
    """
    Plot mossy-fiber input(s) and population firing rates for one or more
    coupled cerebellar cortex (+ optional DCN) mean-field nodes.

    Parameters
    ----------
    t : np.ndarray
        Time vector (assumed in seconds, as in the simulation scripts).

    fmossy_cortex : array (len(t),) OR list of such arrays
        Mossy-fiber drive to the cortex node(s) (GrC/GoC exc_aff).
        Pass a list (one array per node, same order as X_cortex) for a
        multi-node network -> output of find_fixed_point_network().

    X_cortex : array (len(t), 4) OR list of such arrays
        Cortex population traces, columns = [GrC, GoC, MLI, PC].
        Pass a list matching fmossy_cortex for a multi-node network.

    fmossy_DCN : array (len(t),), optional
        Mossy-fiber collateral drive to DCNp. Gets its own input row.
        Only supported together with a single cortex node (no DCN
        support yet for multi-node networks).

    X_DCN : array (len(t), 2), optional
        DCN population traces, columns = [DCNp, DCNi]
        (as returned by find_fixed_point_cortex_DCN_network).

    node_labels : list of str, optional
        One label per cortex node, used as column titles.
        Defaults to None (no titles) for a single node, or "Node 0",
        "Node 1", ... for multiple nodes.

    time_unit : {'ms', 's'}
        Unit for the x-axis. t is assumed to be in seconds and converted
        if time_unit == 'ms' (default).

    figsize : tuple, optional
        Overrides the automatic figure size.

    suptitle : str, optional
        Figure-level title.

    save_path : str, optional
        If given, the figure is saved to this path (e.g. 'traces.png').

    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    fig, axes : matplotlib Figure and 2D array of Axes, shape (n_rows, n_cols).
        axes[-1, :] is the bottom-most row (last input), axes[0, :] is the
        top-most population row (DCNp if present, else PC).
    """

    cortex_inputs = _as_node_list(fmossy_cortex)
    cortex_traces = _as_node_list(X_cortex)
    n_nodes = len(cortex_traces)

    if len(cortex_inputs) != n_nodes:
        raise ValueError("fmossy_cortex and X_cortex must describe the same number of nodes")

    has_dcn = X_DCN is not None
    if has_dcn and n_nodes > 1:
        raise ValueError("DCN plotting is only supported together with a single cortex node")

    # ----- build row order, BOTTOM -> TOP -----------------------------
    # Each entry: (row_label, color, kind)
    rows_bottom_up = [('mfs\n(cortex)', POP_COLORS['input'], 'input_cortex')]
    if has_dcn and fmossy_DCN is not None:
        rows_bottom_up.append(('mfs\nDCNp', POP_COLORS['input'], 'input_DCN'))
    rows_bottom_up += [
        ('GrC', POP_COLORS['GrC'], 'GrC'),
        ('GoC', POP_COLORS['GoC'], 'GoC'),
        ('MLI', POP_COLORS['MLI'], 'MLI'),
        ('PC',  POP_COLORS['PC'],  'PC'),
    ]
    if has_dcn:
        rows_bottom_up += [
            ('DCNi', POP_COLORS['DCNi'], 'DCNi'),
            ('DCNp', POP_COLORS['DCNp'], 'DCNp'),
        ]

    rows_top_down = rows_bottom_up[::-1]
    n_rows = len(rows_top_down)

    if time_unit == 'ms':
        t_plot = np.asarray(t) * 1000.0
        xlabel = 'Time (ms)'
    else:
        t_plot = np.asarray(t)
        xlabel = 'Time (s)'

    if node_labels is None:
        node_labels = [f'Node {i}' for i in range(n_nodes)] if n_nodes > 1 else [None]

    if figsize is None:
        figsize = (5.5 * n_nodes, 1.3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_nodes, figsize=figsize,
                              sharex=True, squeeze=False)

    for col in range(n_nodes):
        Xc = cortex_traces[col]
        fmc = cortex_inputs[col]

        for row, (label, color, kind) in enumerate(rows_top_down):
            ax = axes[row, col]

            if kind == 'input_cortex':
                ax.plot(t_plot, fmc, color=color, lw=1)
            elif kind == 'input_DCN':
                ax.plot(t_plot, fmossy_DCN, color=color, lw=1)
            elif kind == 'GrC':
                ax.plot(t_plot, Xc[:, 0], color=color, lw=1)
            elif kind == 'GoC':
                ax.plot(t_plot, Xc[:, 1], color=color, lw=1)
            elif kind == 'MLI':
                ax.plot(t_plot, Xc[:, 2], color=color, lw=1)
            elif kind == 'PC':
                ax.plot(t_plot, Xc[:, 3], color=color, lw=1)
            elif kind == 'DCNi':
                ax.plot(t_plot, X_DCN[:, 1], color=color, lw=1)
            elif kind == 'DCNp':
                ax.plot(t_plot, X_DCN[:, 0], color=color, lw=1)

            if col == 0:
                ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=14)
            if row == 0 and node_labels[col] is not None:
                ax.set_title(node_labels[col], fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig, axes

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from MF_prediction.load_config_TF import *
    from MF_prediction.theoretical_tools import *
    from MF_prediction.input_library import rect_input
    from params_library.syn_and_connec_library import  *
    from params_library.cell_library import *
    from params_library.params_reformat import *


    # ---- Paths & TF loading (adjust to your environment) ------------------
    root_path = os.getcwd() +'/'
    print(root_path)
    NTWK = 'CRBL_CONFIG_20PARALLEL_wN'
    NTWK_DCN = 'DCN_network'

    FILE_GrC = root_path + '20220519_120033_GrC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.0_fit.npy'
    FILE_GoC = root_path + '20220519_155731_GoC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.3_fit.npy'
    FILE_MLI = root_path + '20220622_085550_MLI_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha1.8_fit.npy'
    FILE_PC  = root_path + '20220622_085610_PC_CRBL_CONFIG_20PARALLEL_wN_tsim5_alpha2.5_fit.npy'

    FILE_DCNi = root_path + '2026_DCNi_alpha0.99.npy'
    FILE_DCNp = root_path + '2026_DCNp_alpha2.152.npy'


    TFgrc = load_transfer_functions('GrC', NTWK, FILE_GrC, alpha=2.0)
    TFgoc = load_transfer_functions_goc('GoC', NTWK, FILE_GoC, alpha=1.3)
    TFmli = load_transfer_functions('MLI', NTWK, FILE_MLI, alpha=5)
    TFpc  = load_transfer_functions('PC',  NTWK, FILE_PC,  alpha=5)

    TFdcn_i = load_transfer_functions_dcni('DCNi', NTWK_DCN, FILE_DCNi, alpha=1)
    TFdcn_p = load_transfer_functions_goc('DCNp', NTWK_DCN, FILE_DCNp, alpha=2.152)

    """
    # JUST TO CHECK THE SIGNATURE
    import inspect
    print('TFdcn_i:', inspect.signature(TFdcn_i))
    print('TFdcn_p:', inspect.signature(TFdcn_p))
    """
    # ---- Simulation parameters --------------------------------------------
    w  = 0
    T  = 3.5e-3
    dt = 1e-4
    t  = np.arange(0, 0.5, dt)

    # Two different mossy-fiber drives
    f_tone1 = rect_input(time=t, t_start=1000, t_end=1500, minval=0, freq=50, noise_freq=0)
    f_tone2 = rect_input(time=t, t_start=2000, t_end=2500, minval=0, freq=50, noise_freq=0)
    f_tone3 = rect_input(time=t, t_start=3000, t_end=3500, minval=0, freq=50, noise_freq=0)

    fmossy_A = np.random.rand(len(t)) * 4 + f_tone1 + f_tone2 + f_tone3

    #fmossy_A = np.random.rand(len(t)) * 4   # node A: 20 Hz background
    #fmossy_B = np.random.rand(len(t)) * 50# node B:  5 Hz background
    #fmossy_C = np.ones_like(fmossy_A)*fmossy_A*2  # node B:  5 Hz background

    # Initial condition set as in Lorenzi et al., 2023, Plos Comp Biol.
    # # TO CHECK: HERE PC BASELINE IS Z+

    M_ctx = get_connectivity_and_synapses_matrix(NAME=NTWK, SI_units=True)
    M_dcn = get_connectivity_and_synapses_matrix(NAME=NTWK_DCN, SI_units=True)

    params_dcni = get_neuron_params('DCNi', SI_units=True)
    reformat_syn_parameters_eglif('DCNi', params_dcni, M_dcn)
    (Qi_dcni, Ti_dcni, Ei_dcni, Gl_dcni, Cm_dcni, El_dcni, Ki_dcni,
     _, _, _, _, _) = pseq_params_dcni(params_dcni)
    print(
        f"""
    Qi_dcni   = {Qi_dcni}
    Ti_dcni   = {Ti_dcni}
    Ei_dcni   = {Ei_dcni}
    Gl_dcni   = {Gl_dcni}
    Cm_dcni   = {Cm_dcni}
    El_dcni   = {El_dcni}
    Ki_dcni   = {Ki_dcni}
    """
    )


    params_dcnp = get_neuron_params('DCNp', SI_units=True)
    reformat_syn_parameters_eglif('DCNp', params_dcnp, M_dcn)
    (Qe_g_dcnp, Qe_m_dcnp, Te_g_dcnp, Te_m_dcnp, Ee_dcnp, Qi_dcnp, Ti_dcnp, Ei_dcnp, Gl_dcnp, Cm_dcnp, El_dcnp, Ke_g_dcnp, Ke_m_dcnp, Ki_dcnp,
     _, _, _, _, _) = pseq_params_eglif_goc(params_dcnp)

    print(
        f"""
    Qe_g_dcnp = {Qe_g_dcnp}
    Qe_m_dcnp = {Qe_m_dcnp}
    Te_g_dcnp = {Te_g_dcnp}
    Te_m_dcnp = {Te_m_dcnp}
    Ee_dcnp   = {Ee_dcnp}
    Qi_dcnp   = {Qi_dcnp}
    Ti_dcnp   = {Ti_dcnp}
    Ei_dcnp   = {Ei_dcnp}
    Gl_dcnp   = {Gl_dcnp}
    Cm_dcnp   = {Cm_dcnp}
    El_dcnp   = {El_dcnp}
    Ke_g_dcnp = {Ke_g_dcnp}
    Ke_m_dcnp = {Ke_m_dcnp}
    Ki_dcnp   = {Ki_dcnp}
    """
    )

    X_DCN = find_fixed_point_DCN(
        TF_DCNp=TFdcn_p, TF_DCNi=TFdcn_i,
        CI_DCN=[5, 5],
        t=t, w=w, T=T,
        fe_mossy=10.0,  # costante
        fe_grc=10.0,  # costante
        fi=1000.0,  # costante
    )

    X_cortex, X_DCN = find_fixed_point_cortex_DCN_network(
        TFgrc, TFgoc, TFmli, TFpc,
        TF_DCNp=TFdcn_p, TF_DCNi=TFdcn_i,
        CI_cortex=[0.5, 5, 15, 38],
        CI_DCN=[5, 5],
        t=t, w=w, T=T,
        fmossy_cortex=fmossy_A,
        fmossy_DCN=fmossy_A,
        w_PC_DCNp= 10,#Ki_dcnp,
        w_PC_DCNi= 1,#Ki_dcni,
        w_GrC_DCNp= 1,#Ke_g_dcnp
    )
    ### OCCHIO che se metto i K qui allora devo metterli anche in mf to input pop!
    ### PER ORA LASCIO 1 PERCHé ESISTE

    # Case 1: single cortex node + DCN (find_fixed_point_cortex_DCN_network)
    plot_coupled_mfm_network(
        t, fmossy_A, X_cortex,
        fmossy_DCN=fmossy_A, X_DCN=X_DCN,
        suptitle='Cortex + DCN network',
        save_path='./test_cortex_dcn_noise.png',
        show=True
    )

    """
    # Case 2: 2-node cortex-only network (find_fixed_point_network)
    X_cortex_B = X_cortex * 0.7 + 2
    plot_coupled_mfm_network(
        t, [fmossy_A, fmossy_A * 0.5], [X_cortex, X_cortex_B],
        node_labels=['Hemisphere R', 'Hemisphere L'],
        suptitle='2-node cortex network (synthetic test)',
        save_path='/home/claude/test_multinode.png',
        show=False
    )
    """

