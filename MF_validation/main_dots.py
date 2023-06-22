import argparse
from CRBL_MF_Model.MF_validation.predictions_analysis import routine_eval_sims

if __name__=='__main__':

    parser = argparse.ArgumentParser(description=
                                     """ 
                                   'Dots-plot of the quantitative score used to evaluate simulation outputs'
                                   """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("SCORE_FILE",
                        help="filename where the scores had been saved")

    args = parser.parse_args()

    MLIw, AUC, Peak, Minim, Min_pause = routine_eval_sims(filename=args.SCORE_FILE)