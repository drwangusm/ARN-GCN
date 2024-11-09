import numpy as np
import argparse, os, time
import h5py

from predict_irn import compute_accuracy

BEST_SCORES = ('/home/mauricio/projects/group-activity/models/Volleyball/'+
    'hp_search/fold_grp+indivs/fusion_att/'+
    'indivs+inter-att-0.25_0.5_0.25-indiv_att-0.25_0.25_0.25-dr_0.25_0/'+
    'y_pred_scores.npy')

GT_PATH = ('/home/mauricio/projects/group-activity/data/datasets/volleyball/'+
           'y_test_true.npy')

#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Fuse video score predictions from two different files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('scores1_filepath',
        help='path to the scores file 1.',
        type=str)
    ap.add_argument('-s','--scores2-filepath',
        help='path to the scores file 2.',
        default=BEST_SCORES,
        type=str)
    ap.add_argument('-o','--output-filepath',
        help='path to the output file to be generated.',
        type=str)
    ap.add_argument('-g','--gt-filepath',
        help='path to the groundtruth classes file.',
        default=GT_PATH,
        type=str)
    ap.add_argument('-w','--weights',
        help="weights",
        default=[1./3, 2./3],
        type=float,
        nargs=2)
    ap.add_argument('-G','--gridsearch',
        help="run gridsearch on weights",
        action='store_true')

    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def fuse_scores_files(scores1_filepath, scores2_filepath, output_filepath=None, 
        weights=[1./3,2./3], gt_filepath=None, gridsearch=False):
    
    print("Fusing scores...")
    print("\t Scores 1 filepath:", scores1_filepath)
    print("\t Scores 2 filepath:", scores2_filepath)
    print("\t Ground Truth filepath:", gt_filepath)
    print("\t Scores output path:", output_filepath)
    print("\t weights:", np.round(weights, 2))
    print("\t Run weights gridsearch:", gridsearch)
    
    scores1 = np.load(scores1_filepath)
    scores2 = np.load(scores2_filepath)
    
    # TODO implement try multiple ranges?
    if not gridsearch:
        # fused_scores = scores1*weights[0] + scores2*weights[1]
        fused_scores = fuse_scores(scores1, scores2, weights=weights)
        
        if output_filepath is not None:
            np.save(output_filepath, fused_scores)
        
        if gt_filepath is not None:
            Y_true_classes = np.load(gt_filepath)
            
            Y_true = np.zeros_like(fused_scores)
            Y_true[np.arange(Y_true_classes.size),Y_true_classes] = 1
            
            acc_1 = compute_accuracy(Y_true, scores1)
            acc_2 = compute_accuracy(Y_true, scores2)
            acc_f = compute_accuracy(Y_true, fused_scores)
            
            print("Acc 1: {:.2%}".format(acc_1))
            print("Acc 2: {:.2%}".format(acc_2))
            print("Acc f: {:.2%}".format(acc_f))
    else:
        grid_weights = [[1./3, 2./3], [0.5, 0.5], [2./3, 1./3]]
        
        if gt_filepath is None:
            raise ValueError("gridsearch option requires gt_filepath!")
        
        Y_true_classes = np.load(gt_filepath)
        
        Y_true = np.zeros_like(scores1)
        Y_true[np.arange(Y_true_classes.size),Y_true_classes] = 1
        
        acc_1 = compute_accuracy(Y_true, scores1)
        acc_2 = compute_accuracy(Y_true, scores2)
        print("Acc 1: {:.2%}".format(acc_1))
        print("Acc 2: {:.2%}".format(acc_2))
        
        max_acc = 0
        
        # grid_results = []
        for weights in grid_weights:
            fused_scores = fuse_scores(scores1, scores2, weights=weights)
            
            acc_f = compute_accuracy(Y_true, fused_scores)
            
            # print("Acc f: {:.2%}".format(acc_f), weights)
            print("Acc f: {:.2%} ({:.2}, {:.2})".format(acc_f, *weights))
            # grid_results.append(acc_f)
            
            if output_filepath is not None and acc_f > max_acc:
                max_acc = acc_f
                np.save(output_filepath, fused_scores)
            
    
    return fused_scores


def fuse_scores(scores1, scores2, weights=[1./3,2./3]):
    fused_scores = scores1*weights[0] + scores2*weights[1]
    return fused_scores
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting fuse_scores - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    fuse_scores_files(**args)

    print('\n> Finished fuse_scores -', time.asctime( time.localtime(time.time()) ))
