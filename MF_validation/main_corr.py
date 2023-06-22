from predictions_analysis import compute_cross_corr, compute_PSD
import numpy as np
import argparse
import statsmodels.api as sm
if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Compute correlation between signals'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("FILE", help="psth file of simulation output")
    args = parser.parse_args()
    Baskbsb, GoCbsb, GrCbsb, PCbsb, Stell = np.load(args.FILE, allow_pickle=True)
    GrCmfm, GoCmfm, MLImfm, PCmfm, mossy = np.load('/Users/robertalorenzi/PycharmProjects/bsb_ws/CRBL_MF_Model/MF_prediction/'+
                     args.FILE.replace('_PSTHvals.npy','.npy'), allow_pickle=True)

    index = np.arange(0, len(GrCmfm), 150)
    PCmfm_binned = PCmfm[index[0:-2]]

    #POWER SPECTRUM ---------------------------------------------------------------------------------------------
    fs = 1/1e-4
    compute_PSD(mossy, fs)

    #CORRELATIONS ----------------------------------------------------------------------------------------------
    corr_PCmfm_PCbsb = compute_cross_corr(PCmfm_binned, PCbsb, 'SNN and MFM output')
    
    #corr_PC_PC = compute_cross_corr(PCmfm, PCmfm, 'PC PC')
    #corr_PC_mossy = compute_cross_corr(PCmfm, mossy, 'PC mossy')