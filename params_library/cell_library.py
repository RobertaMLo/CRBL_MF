"""
Library of single-neurons configurations
"""
from __future__ import print_function


def get_neuron_params(NAME,  SI_units=True, name='', number=1):
    if NAME == 'GrC':
        params = {'name': name, 'N': number,
                  'Gl': 0.2899, 'Cm': 7., 'Trefrac': 1.5, 'tm': 24.15,
                  'El': -62., 'Vthre': -41., 'Vreset': -70., 'delta_v': 0.3,
                  'kadap': 0.022, 'k2': 0.041, 'k1': 0.311, 'A2': -0.94, 'A1': 0.01, 'Ie': -0.888}

    elif NAME == 'GoC':
        params = {'name': name, 'N': number,
                  'Gl': 3.2955, 'Cm': 145., 'Trefrac': 2., 'tm': 44.,
                  'El': -62., 'Vthre': -55., 'Vreset': -75., 'delta_v': 0.4,
                  'kadap': 0.217, 'k2': 0.023, 'k1': 0.031, 'A2': 178.01, 'A1': 259.988, 'Ie': 16.214}

    elif NAME == 'MLI':
        params = {'name': name, 'N': number,
                  'Gl': 1.6, 'Cm': 14.6, 'Trefrac': 1.59, 'tm': 9.125,
                  'El': -68., 'Vthre': -53., 'Vreset': -78., 'delta_v': 1.1,
                  'kadap': 2.025, 'k2': 1.096, 'k1': 1.887, 'A2': 5.863, 'A1': 5.953, 'Ie': 3.711*1.2}

    elif NAME == 'PC':
        params = {'name': name, 'N': number,
                  'Gl': 7.1064, 'Cm': 334., 'Trefrac': 0.5, 'tm': 47.,
                  'El': -59., 'Vthre': -43., 'Vreset': -69., 'delta_v': 3.5,
                  'kadap': 1.491, 'k2': 0.041, 'k1': 0.195, 'A2': 172.622, 'A1': 157.622, 'Ie': 742.54*1.2}

    elif NAME == 'DCNi':
        params = {
            'name': name,
            'N': number,
            'Gl': 1.0,  # g_L: 1e-9 S → nS
            'Cm': 56.0,  # C_m: 5.6e-11 F → pF
            'Trefrac': 3.0,  # t_ref: 0.003 s → ms
            'tm': 56.0,  # derivato: Cm/Gl
            'El': -40.0,  # E_L: -0.04 V → mV
            'Vthre': -39.0,  # V_th: -0.039 V → mV
            'Vreset': -55.0,  # V_reset: -0.055 V → mV
            'delta_v': 3.5,  # delta_V: 0.0035 V → mV
            'kadap': 79.0,  # k_a: 7.9e-08 S → nS
            'k2': 0.044,  # k_2: 44.0 s^-1 → ms^-1
            'k1': 0.041,  # k_1: 41.0 s^-1 → ms^-1
            'A2': 176.358,  # A_2: 1.76358e-10 A → pA
            'A1': 176.358,  # A_1: 1.76358e-10 A → pA
            'Ie': 2.384,  # I_e: 2.384e-12 A → pA
            'Vspike': 20.0,  # V_spike: 0.02 V → mV
            'tauV': 1.0,  # tau_V: 0.001 V → mV (parametro di pendenza dell'escape noise, non un tempo)
            'lambda0': 0.0009  # lambda_0: 0.9 Hz → ms^-1 (kHz)
        }

    elif NAME == 'DCNp':
        params = {
            'name': name,
            'N': number,
            'Gl': 4.3,  # g_L: 4.3e-09 S → nS
            'Cm': 142.0,  # C_m: 1.42e-10 F → pF
            'Trefrac': 1.5,  # t_ref: 0.0015 s → ms
            'tm': 33.023,  # derivato: Cm/Gl = 142/4.3
            'El': -45.0,  # E_L: -0.045 V → mV
            'Vthre': -36.0,  # V_th: -0.036 V → mV
            'Vreset': -55.0,  # V_reset: -0.055 V → mV
            'delta_v': 3.5,  # delta_V: 0.0035 V → mV
            'kadap': 408.0,  # k_a: 4.08e-07 S → nS
            'k2': 0.047,  # k_2: 47.0 s^-1 → ms^-1
            'k1': 0.697,  # k_1: 697.0 s^-1 → ms^-1
            'A2': 3.477,  # A_2: 3.477e-12 A → pA
            'A1': 13.857,  # A_1: 1.3857e-11 A → pA
            'Ie': 75.385,  # I_e: 7.5385e-11 A → pA
            'Vspike': 20.0,  # V_spike: 0.02 V → mV
            'tauV': 3.0,  # tau_V: 0.003 V → mV
            'lambda0': 0.0035  # lambda_0: 3.5 Hz → ms^-1 (kHz)
        }



    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        # print('cell parameters in SI units')
        # MILLI V to V ---------------------------------------------------------------------------------
        params['El'] = 1e-3 * params['El']
        params['Vthre'] = 1e-3 * params['Vthre']
        params['Vreset'] = 1e-3 * params['Vreset']
        params['delta_v'] = 1e-3 * params['delta_v']
        if 'vspike' in params: params['vspike'] = 1e-3 * params['vspike']

        # MILLI s to s -----------------------------------------------------------------------------
        if 'tm' in params: params['tm'] = 1e-3 * params['tm']
        params['Trefrac'] = 1e-3 * params['Trefrac']

        # KILO Hz to Hz --> (ms)^-1 to s^-1 --> (10^-3)^-1 = 10^3 ----------------------------------
        if 'k2' in params: params['k2'] = 1e3 * params['k2']
        if 'k1' in params: params['k1'] = 1e3 * params['k1']

        # (MEGA H)^-1 to H^-1 --> (10^6)^-1 = 10^-6 -------------------------------------------------
        if 'kadap' in params: params['kadap'] = 1e-6 * params['kadap']

        # NANO S to S ------------------------------------------------------------------------------
        if 'a' in params: params['a'] = 1e-9 * params['a']
        params['Gl'] = 1e-9 * params['Gl']

        # PICO A to A -------------------------------------------------------------------------------
        if 'b' in params: params['b'] = 1e-12 * params['b']
        if 'A2' in params: params['A2'] = 1e-12 * params['A2']
        if 'A1' in params: params['A1'] = 1e-12 * params['A1']
        if 'Ie' in params: params['Ie'] = 1e-12 * params['Ie']

        # PICO F to F -------------------------------------------------------------------------------
        params['Cm'] = 1e-12 * params['Cm']
    else:
        print('cell parameters --NOT-- in SI units')

    return params.copy()


if __name__ == '__main__':
    paramsPC = get_neuron_params('PC', True, name='', number=1)
    print(__doc__)