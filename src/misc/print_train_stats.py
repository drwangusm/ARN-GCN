# import numpy as np
import argparse, os, time, glob
import pandas as pd

try:
    from utils import parse_logs
except:
    from misc.utils import parse_logs

def load_args():
    ap = argparse.ArgumentParser(
        description='Print train statistics stored on the csv log files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('summary_dirs',
        help='Directory with the train summary information.',
        nargs='*',
        type=str)
    
    # Optional arguments
    ap.add_argument('-c','--criteria',
        help="criteria for picking the best epoch",
        default='val_accuracy',
        choices=['val_accuracy', 'val_loss', 'val_mean_per_class_accuracy',
                 'val_mAP', 'val_grp_loss', 'val_accuracy_indiv', 'val_mpca_indiv'])
    ap.add_argument('-u','--update',
        help="update summary file (overwrite)",
        action='store_true')
    ap.add_argument('-s','--seqs-eval',
        help="evaluation over sequences average (sample 'all')",
        action='store_true')
        
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_false')
        # action='store_true')
    
    args = ap.parse_args()
    
    return args

def read_runs(summary_dir, criteria=None, seqs_eval=False):
    if not seqs_eval: # Central frames only
        summary_filename = '/summary.csv'
        log_filename = '/training.log'
    else:
        summary_filename = '/summary-pooled_val_accuracy.csv'
        log_filename = '/pooled_val_accuracy.csv'
    
    hist_paths = glob.glob(summary_dir+'/fold_*'+summary_filename)
    hist_paths.sort()
    
    folds_with_summary = [ hist_path.split('/')[-2] for hist_path in hist_paths ]
    
    reruns_dirs = glob.glob(summary_dir+'/rerun_*/') + glob.glob(summary_dir+'/fold_*/')
    reruns_dirs.sort()
    
    reruns_dirs = [ r for r in reruns_dirs 
        if not any( fold in r for fold in folds_with_summary) ]
    
    for rerun_dir in reruns_dirs:
        if 'rerun' in rerun_dir:
            rerun_hist = glob.glob(rerun_dir + log_filename)
            if rerun_hist != []:
                if os.stat(rerun_hist[0]).st_size == 0: # log is empty
                    rerun_hist = []
        else:
            print("WARNING: Training not finished for '{}', using '{}'".format(
                            rerun_dir.split('/')[-2], log_filename))
            fold_reruns_dirs = glob.glob(rerun_dir+'/rerun_*/')
            fold_reruns_dirs.sort()
            
            fold_best_epochs = read_runs(rerun_dir, criteria=criteria, seqs_eval=seqs_eval)
            if fold_best_epochs != []:
                fold_df = pd.concat(fold_best_epochs, axis=1, sort=False).T.reset_index(drop=True)
                max_field = ('val_accuracy' if 'val_accuracy' in fold_df
                             else 'val_mAP')
                max_id = fold_df[max_field].idxmax()
                best_rerun = fold_reruns_dirs[max_id]
                best_hist =  glob.glob(best_rerun + log_filename)
                
                rerun_hist = [ best_hist[0] ]
            else:
                rerun_hist = []
        hist_paths += rerun_hist
        
    hist_paths.sort()
    
    best_epochs = []
    for rerun_idx, hist_path in enumerate(hist_paths):
        hist_df = pd.read_csv(hist_path)
        hist_df = parse_logs(hist_df)
        
        if criteria is None:
            criteria = ('val_accuracy' if 'val_accuracy' in hist_df
                                 else 'val_mAP')
        
        # if 'mean_per_class_accuracy' not in hist_df.columns:
        #     hist_df['mean_per_class_accuracy'] = 0
        #     hist_df['val_mean_per_class_accuracy'] = 0
        
        if criteria.endswith('loss'):
            best_epoch = hist_df.loc[hist_df[criteria].idxmin()]
        # elif criteria.endswith(('accuracy','mAP')):
        elif any([ str in criteria for str in ['accuracy', 'mAP', 'mpca']]):
            if not seqs_eval: # Central frames only
                sorted_hist_df = hist_df.sort_values([criteria, 'val_loss'], 
                    ascending=[False, True])
            else:
                sorted_hist_df = hist_df.sort_values([criteria], ascending=[False])
            best_epoch = sorted_hist_df.iloc[0]
            
        best_epochs.append(best_epoch)
    
    return best_epochs

def pretty_print_stats(stats_df, short_version=False, seqs_eval=False, 
                       eval_metric='acc'):
    acc_tpl = "{:.2%}".format
    mAP_tpl = "{:.3f}".format
    loss_tpl = "{:.3f}".format
    epoc_tpl = "{:.0f}".format
    
    contains_test = any(stats_df.columns.str.startswith('test'))
    
    if eval_metric.startswith('acc'):
        stats_df = stats_df.rename(columns={'accuracy':'acc',
                                    'val_accuracy':'val_acc'})
        print_order = ['epoch','acc','loss','val_acc','val_loss']
        formatters = [epoc_tpl,acc_tpl,loss_tpl,acc_tpl,loss_tpl]
        if contains_test:
            stats_df = stats_df.rename(columns={'test_accuracy':'test_acc'})
            print_order = print_order + ['test_acc']
            formatters = formatters + [acc_tpl]
    elif eval_metric.startswith(('mAP','precision')):
        print_order = ['epoch','mAP','loss','val_mAP','val_loss']
        formatters = [epoc_tpl,mAP_tpl,loss_tpl,mAP_tpl,loss_tpl]
        if contains_test:
            print_order = ['test_mAP'] + print_order
            formatters = [mAP_tpl] + formatters
    
    # if 'mean_per_class_accuracy' in stats_df.columns:
    if False: # Skipping mpca to make output less poluted
        stats_df = stats_df.rename(columns={'mean_per_class_accuracy':'mpca',
                                    'val_mean_per_class_accuracy':'val_mpca'})
        # print_order = ['accuracy','mpca','loss','val_accuracy','val_mpca',
        #                'val_loss','epoch']
        # formatters = [acc_tpl,acc_tpl,loss_tpl,acc_tpl,acc_tpl,loss_tpl,epoc_tpl]

        print_order = print_order + ['mpca','val_mpca']
        formatters = formatters + [acc_tpl, acc_tpl]
    
    if 'accuracy_indiv' in stats_df.columns:
        stats_df = stats_df.rename(columns={'accuracy_indiv':'acc_indiv',
                                    'val_accuracy_indiv':'val_acc_indiv'})
        print_order = print_order + ['acc_indiv','val_acc_indiv']
        formatters = formatters + [acc_tpl, acc_tpl]
    
    # if 'mpca_indiv' in stats_df.columns:
    if False: # Skipping mpca_indiv to make output less poluted
        print_order = print_order + ['mpca_indiv','val_mpca_indiv']
        formatters = formatters + [acc_tpl, acc_tpl]
    
    if short_version:
        print_order = ['val_acc','val_loss']
        stats_df = stats_df[stats_df.index == 'mean']
    
    if seqs_eval:
        print_order = ['val_acc']
    
    # print(stats_df.to_string(columns=print_order, formatters=formatters, 
    print(stats_df[print_order].to_string(formatters=formatters,justify='center'))

def print_train_stats(summary_dir, criteria=None, update=False, seqs_eval=False):
    if not seqs_eval: # Central frames only
        summary_filename = '/summary.csv'
    else:
        summary_filename = '/summary-pooled_val_accuracy.csv'
    
    # if not os.path.exists(summary_dir+summary_filename) and criteria == None:
        # criteria = 'val_accuracy'

    if criteria is None and os.path.exists(summary_dir+summary_filename):
        summary_df = pd.read_csv(summary_dir+summary_filename, index_col=False)
    else:
        best_epochs = read_runs(summary_dir, criteria=criteria, seqs_eval=seqs_eval)
        if best_epochs == []:
            print("ERROR: Unable to read summary.csv or training.log")
            return
        summary_df = pd.concat(best_epochs, axis=1, sort=False).T.reset_index(drop=True)
        
    summary_df.drop('mean', errors='ignore', inplace=True)
    summary_df.drop('std', errors='ignore', inplace=True)
    
    if update:
        summary_df.to_csv(summary_dir+summary_filename)
    
    eval_metric = ('mAP' if 'mAP' in summary_df else 'accuracy')
    # eval_metric = 'mAP'
    # eval_metric = 'accuracy'
    val_eval = 'val_{}'.format(eval_metric)
    max_acc = summary_df.loc[summary_df[val_eval].idxmax()].rename('max_eval')
    min_loss = summary_df.loc[summary_df['val_loss'].idxmin()].rename('min_loss')
    mean = summary_df.mean().rename('mean')
    std = summary_df.std().rename('std')
    # stats = pd.concat([max_acc,mean,std], axis=1).T
    stats = pd.concat([min_loss,max_acc,mean,std], axis=1).T
    
    pretty_print_stats(summary_df, seqs_eval=seqs_eval, eval_metric=eval_metric)
    test_eval = 'test_{}'.format(eval_metric)
    if eval_metric.startswith('acc'):
        fields = [val_eval]
        if test_eval in stats:
            fields.append(test_eval)
        print(stats[fields].T.to_string(float_format="{:.1%}".format,
                                            justify='center'))
    elif eval_metric.startswith('mAP'):
        print(stats[[val_eval]].T.to_string(float_format="{:.3f}".format,
                                            justify='center'))
    
    if criteria is not None and criteria.endswith('loss'):
        print(stats[[criteria]].T.to_string(float_format="{:.4f}".format, 
                                              justify='center', header=False))
    
    # min_loss = summary_df.loc[summary_df['val_loss'].idxmin()].rename('min')
    # mean = summary_df.mean().rename('mean')
    # std = summary_df.std().rename('std')
    # stats = pd.concat([max,mean,std], axis=1).T
    # print(stats[['val_loss']].T.to_string(float_format="{:.4f}".format))

def print_train_stats_all(summary_dirs, **kwargs):
    
    for summary_dir in summary_dirs:
        if summary_dir.endswith('/hp_search/'):
            continue
        print('********** '+summary_dir+' **********')
        print_train_stats(summary_dir, **kwargs)
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())
    
    print('> Starting Print Train Stats - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    print_train_stats_all(**args)

    print('\n> Finished Print Train Stats -', time.asctime( time.localtime(time.time()) ))
