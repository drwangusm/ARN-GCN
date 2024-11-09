import numpy as np
import argparse, sys, os, time, glob

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.metrics import categorical_accuracy, Precision
import tensorflow.keras.backend as K
    
from datasets import Volleyball, ncaa, cad, cad_new
from datasets.data_generator import DataGenerator
from models.irn import get_model, get_fused_model
from models import temporal_irn as tirn
from misc.utils import (read_config, get_persons_info, get_class_weight, 
                        get_fusion_kwargs)

def load_args():
    ap = argparse.ArgumentParser(
        description='Predict using Relational Network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('exp_name',
        help='Experiment/parameter name',
        type=str)
    ap.add_argument('exp_val',
        help='Experiment/parameter value',
        type=str)
    
    # Data arguments
    ap.add_argument('-d','--dataset-name',
        help="dataset to be used for predicting",
        default='Volleyball',
        choices=['Volleyball', 'ncaa', 'cad', 'cad-new'])
    ap.add_argument('-f','--fold',
        help="dataset fold to be used for predicting",
        default='grp')
    ap.add_argument('-s','--split',
        help="split of data to use",
        default='test',
        choices=['validation', 'test', 'train'])
    ap.add_argument('-r','--rerun',
        help="Rerun id to use (-1 for all)",
        default=-1,
        type=int)
    ap.add_argument('-c','--criteria',
        help="criteria for chosing weights",
        default='eval',
        choices=['eval', 'loss'])
    ap.add_argument('-t','--temp-irn',
        help="use temporal_irn model",
        action='store_true')
    ap.add_argument('-F','--fusion-irn',
        help="use fusion_irn model",
        action='store_true')
    ap.add_argument('-z','--split-fold',
        help="fold number to run",
        type=int,
        default=0)
    
    # Predicting arguments
    ap.add_argument('-b', '--batch-size',
        type=int,
        default=96,
        help='batch size to use during predction')
    ap.add_argument('-v','--verbose',
        help="verbose level",
        type=int,
        default=1)
    ap.add_argument('-S','--save-preds',
        help="Save predicted and true classes",
        action='store_true')
    ap.add_argument('-P','--save-scrs',
        help="Save predicted probability scores",
        action='store_true')
    ap.add_argument('--noise-std',
        type=float,
        default=0.,
        help='add gaussian noise to data, with entered std (cm)')
    ap.add_argument('--noise-dr',
        type=float,
        default=0.,
        help='add dropout noise to data, with entered rate')
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def predict_model(model, data, use_data_gen=True, verbose=2, 
                  use_multiproc=False):
    if use_data_gen:
        data_generator = data
        
        ### Debugging options
        # data_generator.num_batches = 4
        # use_multiproc = True
        # use_multiproc = False
        # if not use_multiproc:
        #     print("Warning: Not using multiprocessing!")
        
        # print("Using predict_generator()...")
        
        if not isinstance(model.output, list):
            num_outputs = 1
            Y_true = []
        else:
            num_outputs = len(model.output)
            Y_true = [[] for _ in range(num_outputs)]
            
        for batch_idx in range(len(data_generator)):
            batch_y = data_generator.get_y(batch_idx)
            # data_generator.get_sample_weight(batch_y, batch_gt)
            if num_outputs == 1:
                Y_true += batch_y.tolist()
            else:
                for out_idx in range(num_outputs):
                    Y_true[out_idx] += batch_y[out_idx].tolist()
        
        ### TODO Using tf.data
        # def generator():
        #     for batch_idx in range(len(data_generator)):
        #         batch_x, *batch_y = data_generator[batch_idx]
        #         yield (tuple(batch_x),) + tuple(batch_y)
        # sample_x, *sample_y = data_generator[0]
        
        # output_types_x = tuple([tf.as_dtype(np.asarray(sample_x).dtype)]*len(sample_x))
        # output_types_y = tuple([ tf.as_dtype(np.asarray(out).dtype) for out in sample_y ])
        # output_types = (output_types_x,) + output_types_y
        # dataset = tf.data.Dataset.from_generator(generator, output_types)
        
        # Y_pred = model.predict(dataset, 
        Y_pred = model.predict(data_generator, 
              max_queue_size=10,workers=5, use_multiprocessing=use_multiproc, 
              verbose=verbose)
        
        ## Running my own loop with predict_on_batch()
        ## Former workaround when order got mixed up
        # print("Using predict_on_batch()...")
        # Y_pred = []
        # Y_true = []
        # if verbose > 0: progbar = ProgressBar(max_value=len(data_generator))
        # for batch_idx in range(len(data_generator)):
        #     if verbose > 0: progbar.update(batch_idx)
        #     batch_x, batch_y = data_generator[batch_idx]
        #     Y_pred += list(model.predict_on_batch(batch_x))
        #     Y_true += batch_y.tolist()
        # if verbose > 0: progbar.finish()
        
    else:
        raise NotImplementedError("Parameter 'use_data_gen' with 'False'")
        # X = data
        # reshaped_Y_pred = model.predict(X, batch_size=batch_size, 
        #     verbose=verbose)
    
    ### TODO remove invalid rows from indiv?
    
    return Y_pred, Y_true

def predict_irn(weights_path, model_kwargs, data_kwargs, dataset_name, 
        dataset_fold=None, batch_size=60, verbose=2, use_data_gen=True, 
        split='test', eval_metric='accuracy', temp_irn=False,
        fusion_irn=False, fusion_kwargs={}, append_log=True):
    if verbose > 1:
        print("***** Predicting parameters *****")
        print("\t weights_path:", weights_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Dataset split:", split)
        print("\t Skeleton info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t Model info")
        if not fusion_irn:
            print("\t Model info")
            for key, value in model_kwargs.items():
                print("\t > {}: {}".format(key, value))
        else:
            print("\t Models info")
            for submodel_kwargs in model_kwargs:
                print("\t Model:", submodel_kwargs['rel_type'])
                for key, value in submodel_kwargs.items():
                    print("\t > {}: {}".format(key, value))
            print("\t Fusion info")
            for key, value in fusion_kwargs.items():
                print("\t > {}: {}".format(key, value))
        # print("\t Relation info")
        # print("\t > relationship type:", rel_type)
        # print("\t > fusion type:", fuse_type)
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
        print("\t > Use data genarator:", use_data_gen)
        print("\t > Evaluation metric:", eval_metric)
    
    if dataset_name == 'Volleyball':
        dataset = Volleyball
    elif dataset_name == 'ncaa':
        dataset = ncaa
    elif dataset_name == 'cad':
        dataset = cad
    elif dataset_name == 'cad-new':
        dataset = cad_new
    else:
        print("ERROR: Invalid dataset -", dataset_name)
        return
    
    if verbose > 0:
        print("Reading data...")
    
    if not fusion_irn:
        indiv_out = model_kwargs.get('indiv_out', False)
        grp_out = model_kwargs.get('grp_out', True)
    else:
        indiv_out = model_kwargs[0].get('indiv_out', False)
        grp_out = model_kwargs[0].get('grp_out', True)
    
    if indiv_out and not grp_out:
        raise NotImplementedError()
    
    ### Preparing data
    if use_data_gen:
        if verbose > 0:
            print("> Using DataGenerator")
        # TO-DO if temp_irn, use DataGeneratorSeq
        split_generator = DataGenerator(dataset_name, dataset_fold, split,
            batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            indiv_out=indiv_out, use_cw=False,
            **data_kwargs)
        batch_data = split_generator[0]
        X, Y_true = batch_data[:2]
        
        split_data = split_generator
    else:
        raise NotImplementedError("Parameter 'use_data_gen' with 'False'")
        if verbose > 0:
            print("> Reading all data at once")
        if split == 'train':
            X, Y_true = dataset.get_train(dataset_fold, **data_kwargs)
        elif split == 'validation':
            X, Y_true = dataset.get_val(dataset_fold, **data_kwargs)
        elif split == 'test':
            X, Y_true = dataset.get_test(dataset_fold, **data_kwargs)
        
        split_data = [X, Y_true]
    
    ### Creating model
    if verbose > 0:
        print("Creating model...")
    
    selected_joints = data_kwargs.get('selected_joints')
    num_joints = (len(selected_joints) if selected_joints is not None else 25)
        
    timesteps = data_kwargs['timesteps']
    add_joint_idx = data_kwargs['add_joint_idx']
    add_body_part = data_kwargs['add_body_part']
    overhead = add_joint_idx + add_body_part # True/False = 1/0
    object_shape = (len(X[0][0]),)
    num_dim = (object_shape[0]-overhead)//timesteps
    
    # Num indivs info missing from model_kwargs
    if not fusion_irn:
        if 'num_indivs' not in model_kwargs and 'num_indivs' in data_kwargs:
            model_kwargs['num_indivs'] = data_kwargs['num_indivs']
    else:
        for sub_model_kwargs in model_kwargs:
            if 'num_indivs' not in sub_model_kwargs and 'num_indivs' in data_kwargs:
                sub_model_kwargs['num_indivs'] = data_kwargs['num_indivs']
    
    if not indiv_out:
        output_size = len(Y_true[0])
        indiv_output_size = 0
    else:
        output_size = len(Y_true[0][0])
        indiv_output_size = len(Y_true[1][0])
    
    if not temp_irn:
        if not fusion_irn:
            model = get_model(num_objs=num_joints, object_shape=object_shape, 
                output_size=output_size, num_dim=num_dim, overhead=overhead, 
                indiv_output_size=indiv_output_size, **model_kwargs)
        else:
            model = get_fused_model(selected_joints, object_shape=object_shape,
                output_size=output_size, num_dim=num_dim, overhead=overhead,
                models_kwargs=model_kwargs, indiv_output_size=indiv_output_size,
                **fusion_kwargs)
            
    else:
        seq_len = None
        model = tirn.get_model(num_objs=num_joints, object_shape=object_shape, 
            output_size=output_size, num_dim=num_dim, overhead=overhead,
            seq_len=seq_len, **model_kwargs)
    
    if len(model.input) != len(X):
        print("Warning: Model input size differs from data size.")
        print("\t Model:", len(model.input), "Data:", len(X))
    
    if not isinstance(weights_path, list):
        weights_path = [weights_path]
    
    ### Predicting
    best_result = 0
    results = []
    for filepath in weights_path:
        model.load_weights(filepath)
        
        if verbose > 0:
            print("\nWeights loaded:", filepath)
            print("Starting predicting...")
        
        Y_pred, Y_true = predict_model(model, split_generator, 
                                   use_data_gen=use_data_gen, verbose=verbose,
                                   use_multiproc=True)
        
        ### Verbosing eval metrics
        if verbose > 0:
            if eval_metric =='accuracy':
                group_indivs = data_kwargs.get('group_indivs', False)
                
                if not indiv_out:
                    padded_acc = compute_accuracy(Y_true, Y_pred)
                    
                    # split_gt is only necessary for individuals prediction
                    # Because for some clips the number of indivs had to be padded
                    # So these padded rows has to discarded from the eval metric
                    if group_indivs:
                        true_acc = padded_acc
                        print("Split '{}' - acc: {:.2%}".format(split, true_acc))
                    else:
                        split_gt = dataset.get_split_gt(split, dataset_fold)
                        true_acc = compute_accuracy(Y_true, Y_pred, ref_gt=split_gt)
                        
                        tpl = "Split '{}' - True acc: {:.2%} | Padded acc: {:.2%}"
                        print(tpl.format(split, true_acc, padded_acc))
                else:
                    Y_pred_grp, *Y_pred_inds = Y_pred
                    Y_true_grp, *Y_true_inds = Y_true
                    
                    num_indivs = len(Y_true_inds)
                    
                    # Stack and reshape to keep players from same video in seq.
                    Y_pred_indiv = np.stack(Y_pred_inds, 1).reshape(
                                                       (-1, indiv_output_size))
                    Y_true_indiv = np.stack(Y_true_inds, 1).reshape(
                                                       (-1, indiv_output_size))
                    
                    # Y_pred_indiv = np.concatenate(Y_pred_inds, axis=0)
                    # Y_true_indiv = np.concatenate(Y_true_inds, axis=0)
                    
                    acc_grp = compute_accuracy(Y_true_grp, Y_pred_grp)
                    
                    split_gt = dataset.get_split_gt(split, dataset_fold)
                    split_gt = split_gt.head(len(Y_true_grp)) # work around when subsampling
                    acc_indiv = compute_accuracy(Y_true_indiv, Y_pred_indiv, 
                                     ref_gt=split_gt, num_indivs=num_indivs)
                    # cw = get_class_weight(dataset, indiv_actions=True)
                    cw = get_class_weight(dataset, indiv_actions=True, 
                                          dataset_fold=dataset_fold)
                    mpca_indiv = compute_accuracy(Y_true_indiv, Y_pred_indiv, 
                       ref_gt=split_gt, num_indivs=num_indivs, class_weight=cw)
                    tpl = "Split '{}' - group acc: {:.2%} - indivs acc: {:.2%}"
                    print(tpl.format(split, acc_grp, acc_indiv))
                    print("\t\t\t\tindivs mpca: {:.2%}".format(mpca_indiv))
                    
                    true_acc = acc_grp
                
                results.append(true_acc)
                
                # class_weight = get_class_weight(dataset, 
                #                                 indiv_actions=(not group_indivs))
                # mean_per_class_acc = compute_accuracy(Y_true, Y_pred, 
                #                   ref_gt=split_gt, class_weight=class_weight)
                # print("\tMean per class acc: {:.2%}".format(mean_per_class_acc))
            elif eval_metric == 'precision':
                prec, precs_per_class = compute_precision(Y_true, Y_pred, 
                                                       per_class=True)
                results.append(prec)
                
                print("Split {} - mAP: {:.3f}".format(split, prec))
                for class_idx, cl_prec in enumerate(precs_per_class):
                    # print("> class {}: {:.3f}".format(class_idx, prec))
                    print("> class {:>2d}: {:.3f} ({})".format(class_idx, cl_prec,
                                               dataset.GRP_ACTIVITIES[class_idx]))
            
            if results[-1] > best_result:
                print("Updating best predicitons...")
                best_result = results[-1]
                best_Y_pred = Y_pred
    
    if best_result == 0: # When verbose is 0, so it is only one weight
        best_Y_pred = Y_pred
    
    if verbose > 0:
        print("Summary:")
        for filepath, result in zip(weights_path, results):
            weight_dirpath, weight_file = os.path.split(filepath)
            _, rerun = os.path.split(weight_dirpath)
            
            res_format = ("{:.3f}".format if eval_metric == 'precision' 
                          else "{:.1%}".format)
            # print("\t", res_format(result), "=", weight_name)
            weight_type = ('best val '+eval_metric if '-' in weight_file 
                           else 'best val loss')
            print("\t", res_format(result), "=", rerun, weight_type)
    
    if append_log and split == 'test':
        test_eval = ('test_mAP' if eval_metric == 'precision' else 'test_accuracy')
        reruns = set([ os.path.basename(os.path.dirname(p)) for p in weights_path])
        reruns = list(reruns)
        for rerun in reruns:
            rerun_weights = [ p for p in weights_path if rerun in p]
            rerun_results = [ results[weights_path.index(p)] for p in rerun_weights]
            rerun_folder = os.path.dirname(rerun_weights[0])
            log_filepath = os.path.join(rerun_folder, 'training.log')
            
            header = np.genfromtxt(log_filepath, dtype=str, delimiter=',', 
                                   max_rows=1)
            values = np.genfromtxt(log_filepath, dtype=float, delimiter=',', 
                                   skip_header=1)
            
            if test_eval not in header:
                max_test = max(rerun_results)
                test_by_loss, test_by_eval = 0, 0
                for rerun_weight,rerun_result in zip(rerun_weights,rerun_results):
                    weight_file = os.path.basename(rerun_weight)
                    if '-' in weight_file:
                        test_by_eval = rerun_result
                    else:
                        test_by_loss = rerun_result
                
                num_epochs = values.shape[0]
                test_values = [[max_test, test_by_loss, test_by_eval]]*num_epochs
                
                test_header = [test_eval, test_eval[:8]+'_by_loss', 
                               test_eval[:8]+'_by_eval']
                
                new_values = np.concatenate((values, test_values), axis=-1)
                new_header = ','.join(np.concatenate((header, test_header)))
                
                os.rename(log_filepath, log_filepath+'.bkp')
                np.savetxt(log_filepath, new_values, fmt='%g', delimiter=',',
                            header=new_header, comments='')
                print("Log updated:", log_filepath)
    
    
    return best_Y_pred, Y_true

def exp_predict_irn(dataset_name, fold, exp_name, exp_val, rerun=-1, 
                    criteria='eval', temp_irn=False, fusion_irn=False, 
                    save_preds=False, save_scrs=False, split_fold=0,
                    noise_std=0, noise_dr=0, 
                    **predict_kwargs):
    print("***** Predict parameters *****")
    print("\t Dataset:", dataset_name)
    print("\t Dataset fold:", fold)
    print("\t Dataset Split fold:", split_fold)
    print("\t Experiment:", exp_name+"/"+exp_val)
    print("\t Using Temporal IRN:", temp_irn)
    print("\t Using Fusion IRN:", fusion_irn)
    
    base_path = 'models/{}/hp_search/fold_{}/{}/{}'.format(
    # base_path = '../models/{}/hp_search/fold_{}/{}/{}'.format(
        dataset_name, fold, exp_name, exp_val)
    base_path = os.path.join(*base_path.split('/'))
    config_filepath = os.path.join(base_path, 'parameters.cfg')
    
    print("\t base_path:")
    print("\t\t", base_path)
    
    if not fusion_irn:
        data_kwargs, model_kwargs, train_kwargs = read_config(config_filepath)
        fusion_kwargs = None
    else:
        fusion_kwargs, train_kwargs = read_config(config_filepath, fusion=True)
        config_filepaths = fusion_kwargs.pop('config_filepaths')
        data_kwargs, models_kwargs = get_fusion_kwargs(config_filepaths)
        model_kwargs = models_kwargs
    
    eval_metric = train_kwargs.get('eval_metric', 'accuracy')
    predict_kwargs['eval_metric'] = eval_metric
    
    print("\t Noise info:", fusion_irn)
    print("\t > noise_std: {}".format(noise_std))
    print("\t > noise_dr: {}".format(noise_dr))
    
    if noise_std > 0:
        data_kwargs['noise_std'] = noise_std
    if noise_dr > 0:
        data_kwargs['noise_dr'] = noise_dr
    
    if rerun >= 0:
        rerun_dir = 'rerun_{}'.format(rerun)
        if criteria == 'eval':
            weights_file = ('relnet_weights-val_acc.hdf5' 
                            if eval_metric == 'accuracy'
                            else 'relnet_weights-val_prec.hdf5')
        elif criteria == 'loss':
            weights_file = 'relnet_weights.hdf5'
        weights_path = [os.path.join(base_path, rerun_dir, weights_file)]
        
        print("\t Weights file:", rerun_dir+'/'+weights_file)
    else:
        weights_path = glob.glob(base_path+'/rerun_*/relnet_weights*.hdf5')
        weights_path = sorted(weights_path)
        print("\t Weights files to run predict with:")
        for filepath in weights_path:
            print(">", filepath)
    
    print("\t Predict info")
    for key, value in predict_kwargs.items():
        print("\t > {}: {}".format(key, value))
    
    Y_pred, Y_true = predict_irn(weights_path, model_kwargs, data_kwargs, 
          # dataset_name, fold, temp_irn=temp_irn, fusion_irn=fusion_irn,
          dataset_name, split_fold, temp_irn=temp_irn, fusion_irn=fusion_irn,
          fusion_kwargs=fusion_kwargs, **predict_kwargs)
    
    if save_preds:
        if not fusion_irn:
            indiv_out = model_kwargs.get('indiv_out', False)
        else:
            indiv_out = model_kwargs[0].get('indiv_out', False)
        
        save_predictions(Y_pred, Y_true, base_path, indiv_out=indiv_out)
        print("Predictions and true classes saved.")
    
    if save_scrs:
        if not fusion_irn:
            indiv_out = model_kwargs.get('indiv_out', False)
        else:
            indiv_out = model_kwargs[0].get('indiv_out', False)
        
        save_scores(Y_pred, base_path, indiv_out=indiv_out)
        print("Predictions scores saved.")

def save_predictions(Y_pred, Y_true, out_path, indiv_out=False):
    if not indiv_out:
        # invert one-hot and save as ints
        classes_true = np.argmax(Y_true, axis=1).astype('int8')
        classes_pred = np.argmax(Y_pred, axis=1).astype('int8')
        
        np.save(os.path.join(out_path, 'y_true.npy'), classes_true)
        np.save(os.path.join(out_path, 'y_pred.npy'), classes_pred)
    else:
        ### TODO indiv_out
        Y_pred_grp, *Y_pred_inds = Y_pred
        Y_true_grp, *Y_true_inds = Y_true
        
        # num_indivs = len(Y_true_inds)
        indiv_output_size = np.shape(Y_pred_inds)[-1]
        ### TODO remove invalid rows from indiv?
        
        # Stack and reshape to keep players from same video in seq.
        Y_pred_indiv = np.stack(Y_pred_inds, 1).reshape(
                                           (-1, indiv_output_size))
        Y_true_indiv = np.stack(Y_true_inds, 1).reshape(
                                           (-1, indiv_output_size))
        
        classes_true = np.argmax(Y_true_grp, axis=1).astype('int8')
        classes_pred = np.argmax(Y_pred_grp, axis=1).astype('int8')
        np.save(os.path.join(out_path, 'y_true.npy'), classes_true)
        np.save(os.path.join(out_path, 'y_pred.npy'), classes_pred)
        
        classes_true = np.argmax(Y_true_indiv, axis=1).astype('int8')
        classes_pred = np.argmax(Y_pred_indiv, axis=1).astype('int8')
        np.save(os.path.join(out_path, 'y_true_indiv.npy'), classes_true)
        np.save(os.path.join(out_path, 'y_pred_indiv.npy'), classes_pred)

def save_scores(Y, out_path, indiv_out=False):
    if not indiv_out:
        # invert one-hot and save as ints
        # classes_pred = np.argmax(Y_pred, axis=1).astype('int8')
        # np.save(os.path.join(out_path, 'y_pred_scores.npy'), classes_pred)
        
        np.save(os.path.join(out_path, 'y_pred_scores.npy'), Y)
    else:
        ### TODO indiv_out
        Y_grp, *Y_inds = Y
        
        # num_indivs = len(Y_true_inds)
        indiv_output_size = np.shape(Y_inds)[-1]
        ### TODO remove invalid rows from indiv?
        
        # Stack and reshape to keep players from same video in seq.
        Y_indiv = np.stack(Y_inds, 1).reshape((-1, indiv_output_size))
        
        # classes_pred = np.argmax(Y_pred_grp, axis=1).astype('int8')
        # np.save(os.path.join(out_path, 'y_pred_scores.npy'), classes_pred)
        np.save(os.path.join(out_path, 'y_pred_scores.npy'), Y_grp)
        
        # classes_pred = np.argmax(Y_pred_indiv, axis=1).astype('int8')
        # np.save(os.path.join(out_path, 'y_pred_scores_indiv.npy'), classes_pred)
        np.save(os.path.join(out_path, 'y_pred_scores_indiv.npy'), Y_indiv)

def compute_precision_tf(Y_true, Y_pred, per_class=False, num_classes=11, 
                         thresholds=None, binarize=False):
    # Problem with this approach: overall precision is not the mean per class
    
    bin_Y_pred = np.zeros_like(Y_pred)
    max_idxs = Y_pred.argmax(axis=1)
    bin_Y_pred[np.arange(len(Y_pred)), max_idxs] = 1.
    Y_pred = bin_Y_pred
    
    m = Precision(thresholds=thresholds)
    m.update_state(Y_true, Y_pred)
    prec = m.result().numpy()
    
    if per_class:
        precs_per_class = []
        for class_id in range(num_classes):
            m = Precision(class_id=class_id, thresholds=thresholds)
            m.update_state(Y_true, Y_pred)
            precs_per_class.append(m.result().numpy())
    
    if not per_class:
        return prec
    else:
        return prec, precs_per_class

def compute_precision_sk(Y_true, Y_pred, per_class=False, num_classes=11):
    from sklearn.metrics import classification_report
    
    ### TODO Implement threshold?
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    class_rep = classification_report(y_true, y_pred, output_dict=True)
    prec = class_rep['macro avg']['precision']
    
    if per_class:
        precs_per_class = []
        for class_id in range(num_classes):
            precs_per_class.append(class_rep[str(class_id)]['precision'])
    
    if not per_class:
        return prec
    else:
        return prec, precs_per_class

def compute_precision(Y_true, Y_pred, per_class=False, num_classes=11):
    return compute_precision_sk(Y_true, Y_pred, 
         per_class=per_class, num_classes=num_classes)
    
def compute_accuracy(Y_true, Y_pred, ref_gt=None, class_weight=None, 
                     num_indivs=None):
    """
    Compute accuracy using true and predicted labels.
    If reference groundtruth (ref_gt) is provided, it will use this 
    information to prune Y_true and Y_pred of possible padded rows.

    Parameters
    ----------
    Y_true : array
        True categorical labels from data.
    Y_pred : array
        Predicted categorical labels from data.
    ref_gt : pandas.Dataframe, optional
        Reference groundtruth for prunning padded rows. The default is None.

    Returns
    -------
    acc : float
        Categorical accuracy.
    """
    
    if ref_gt is not None:
        max_num_persons = len(get_persons_info(ref_gt.iloc[0],
                                               prune_missing=False))
        pruned_idx = []
        for video_idx, video_gt in ref_gt.iterrows():
            num_persons = get_persons_info(video_gt).shape[0]
            video_mask = [True] * num_persons 
            video_mask += [False] * (max_num_persons-num_persons)
            
            if num_indivs is not None:
                video_mask = video_mask[:num_indivs]
            
            pruned_idx += video_mask
        
        Y_pred = np.asarray(Y_pred)[pruned_idx]
        Y_true = np.asarray(Y_true)[pruned_idx]
    
    acc_tensor = categorical_accuracy(Y_true, Y_pred)
    hits = K.eval(acc_tensor)
    if class_weight is not None:
        Y_classes = np.argmax(Y_true, axis=1)
        class_accs = []
        for key, value in class_weight.items():
            class_accs.append(hits[Y_classes==key].mean())
            hits[Y_classes==key] *= value
    acc = hits.mean()
    
    return acc

#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting Predict RN - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    exp_predict_irn(**args)

    print('\n> Finished Predict RN -', time.asctime( time.localtime(time.time()) ))
