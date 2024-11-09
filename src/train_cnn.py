import numpy as np
import argparse, os, time

import tensorflow as tf
# if int(tf.__version__.split('.')[1]) >= 14:
#     tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datasets import Volleyball
from datasets.data_generator_video import DataGenerator
from models.cnn_grp import get_model#, get_fused_model
# from models import temporal_irn as tirn
from misc.utils import read_config, get_class_weight
from predict_irn import predict_model, compute_accuracy, compute_precision
from predict import save_scores
# from train_irn import train_cnn

def load_args():
    ap = argparse.ArgumentParser(
        description='Train CNN.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('output_path',
        help='path to output the model snapshots and log',
        type=str)
    
    # Data arguments
    ap.add_argument('-d','--dataset-name',
        help="dataset to be used for training",
        default='Volleyball',
        choices=['Volleyball', 'ncaa', 'cad', 'cad-new'])
    ap.add_argument('-f','--dataset-fold',
        help="dataset fold to be used for training",
        default=9,
        type=int)
    ap.add_argument('-t', '--timesteps',
        type=int,
        default=10,
        help='how many timesteps to use')
    
    # Training arguments
    ap.add_argument('-l', '--learning-rate',
        type=float,
        default=1e-4,
        help='learning rate for training')
    ap.add_argument('-r', '--drop-rate',
        type=float,
        default=0.3,
        help='dropout rate for training')
    ap.add_argument('-b', '--batch-size',
        type=int,
        default=32,
        help='batch size used to train')
    ap.add_argument('-G', '--gpus',
        type=int,
        default=1,
        help='number of gpus to use')
    ap.add_argument('-e', '--epochs',
        type=int,
        default=20,
        help='number of epochs to train')
    ap.add_argument('-c', '--checkpoint-period',
        type=int,
        default=100,
        help='interval (number of epochs) between checkpoints')
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

"""
def set_callbacks(output_path, checkpoint_period, batch_size, 
                use_earlystopping=True, use_lr_sched=False, lr=1e-4, 
                eval_metric='accuracy', epoch_verbose=False, indiv_out=False):


def train_model(model, verbose, learning_rate, output_path, checkpoint_period, 
        batch_size, epochs, use_data_gen, train_data, val_data, subsample_ratio,
        use_earlystopping=True, use_lr_sched=False, class_weight=None, 
        metric_class_weight=None, eval_metric='accuracy', epoch_verbose=False,
        validation_steps=None, grp_out=True, indiv_out=False, indiv_loss_mult=1):
    
"""

def train_cnn(output_path, dataset_name, model_kwargs, data_kwargs,
        dataset_fold=None, drop_rate=0.1, batch_size=60, epochs=100, 
        checkpoint_period=5, learning_rate=1e-4, 
        temp_cnn=False,
        subsample_ratio=None, gpus=1, verbose=2, 
        use_data_gen=True, use_earlystopping=True, use_lr_sched= False, 
        evaluate_on_test=False, eval_metric='accuracy', skips_val=False,
        reshuffle_train=True, fusion_cnn=False, fusion_kwargs={},
        epoch_verbose=False, use_cw=True, jitter_timesteps=False,
        jitter_timesteps_mag=1, 
        indiv_loss_mult=1, ind_clip_weights=None, shuffle_sides=False,
        save_preds=False):
    if verbose == 3:
        epoch_verbose = True
        verbose = 0
    
    if verbose > 0:
        epoch_verbose = False
        print("***** Training parameters *****")
        print("\t Output path:", output_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Data info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
        
        if not fusion_cnn:
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
        # print("\t > Temporal irn:", temp_cnn)
        
        print("\t Training options")
        print("\t > Batch Size:", batch_size)
        print("\t > Epochs:", epochs)
        print("\t > Checkpoint Period:", checkpoint_period)
        print("\t > Learning Rate:", learning_rate)
        print("\t > Dropout rate:", drop_rate)
        print("\t > Training Subsample Ratio:", subsample_ratio)
        print("\t > use_earlystopping:", use_earlystopping)
        print("\t > use_lr_sched:", use_lr_sched)
        print("\t > eval_metric:", eval_metric)
        print("\t > evaluate_on_test:", evaluate_on_test)
        print("\t > Use class weights (indivs only):", use_cw)
        print("\t > indiv_loss_mult:", indiv_loss_mult)
        print("\t > ind_clip_weights:", ind_clip_weights)
        print("\t > shuffle_sides:", shuffle_sides)
        print("\t > jitter_timesteps:", jitter_timesteps, 
              'mag:', jitter_timesteps_mag)
    
    if dataset_name == 'Volleyball':
        dataset = Volleyball
        # print("Warning: Setting shuffle_sides as True")
        # shuffle_sides = True
    else:
        print("ERROR: Invalid dataset -", dataset_name)
        return
    
    if verbose > 0:
        print("Reading data...")
    
    # sort_players = data_kwargs.get('sort_players', False)
    if not fusion_cnn:
        indiv_out = model_kwargs.get('indiv_out', False)
        grp_out = model_kwargs.get('grp_out', True)
        # rel_type = model_kwargs['rel_type']
        # if rel_type.startswith('inter-graph') and not sort_players:
        #     print("Warning: rel_type is inter-graph but sort_players is False.")
    else:
        indiv_out = fusion_kwargs.get('indiv_out', False)
        grp_out = fusion_kwargs.get('grp_out', True)
        # indiv_conc_pool = fusion_kwargs.get('indiv_conc_pool', False)
        
        # require_sort_players = np.any([ m['rel_type'].startswith('inter-graph')
        #                                for m in model_kwargs])
        # if require_sort_players and not sort_players:
        #     print("Warning: rel_type is inter-graph but sort_players is False.")
    
    cnn_arch = model_kwargs['cnn_arch']
    default_img_size = ((224,224) if cnn_arch != 'inception-v3' else (299,299))
    data_kwargs.setdefault('img_size', default_img_size)
    
    # print("WARNING: Hard-coded dataset fold set to 0.")
    # dataset_fold = 0
    #### Peparing Generators
    if use_data_gen:
        if verbose > 0:
            print("> Using DataGenerator")
        # print("WARNING: Setting train shuffle_indiv_order to 'True'.")
        # print("WARNING: Setting train shuffle_sides to 'True'.")
        # jitter_timesteps = True
        if not shuffle_sides:
            print("WARNING: shuffle_sides is 'False'!")
        if jitter_timesteps:
            print("WARNING: Setting train jitter_timesteps to 'True'.")
        if not temp_cnn:
            train_generator = DataGenerator(dataset_name, dataset_fold, 'train',
                batch_size=batch_size, reshuffle=reshuffle_train, 
                shuffle_indiv_order=False, shuffle_sides=shuffle_sides,
                jitter_timesteps=jitter_timesteps, 
                jitter_timesteps_mag=jitter_timesteps_mag, 
                indiv_out=indiv_out, ind_clip_weights=ind_clip_weights, 
                grp_out=grp_out,
                use_cw=use_cw, **data_kwargs)
            val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
                indiv_out=indiv_out, grp_out=grp_out, **data_kwargs)
        else:
            # print("WARNING: Setting pad_sequences as 'False'.")
            raise NotImplementedError("temp_cnn = True")
            print("WARNING: Missing grp+indivs update, and potentially more.")
            # train_generator = DataGeneratorSeq(dataset_name, dataset_fold,'train',
            #     batch_size=batch_size, reshuffle=reshuffle_train,
            #     shuffle_indiv_order=False, 
            #     pad_sequences=True, buffer_data=False, **data_kwargs)
            # val_generator=DataGeneratorSeq(dataset_name, dataset_fold, 'validation',
            #     batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            #     pad_sequences=True, buffer_data=False, **data_kwargs)
            
        # X_train, Y_train = train_generator[0]
        # X_val, Y_val = val_generator[0]
        
        batch_data = train_generator[0]
        X_train, Y_train = batch_data[:2]
        batch_data = val_generator[0]
        X_val, Y_val = batch_data[:2]
        
        # print("X_train.shape:", np.asarray(X_train).shape)
        
        train_data = train_generator
        val_data = val_generator
    else:
        raise NotImplementedError("Parameter 'use_data_gen' with 'False'")
        if verbose > 0:
            print("> Reading all data at once")
        X_train, Y_train = dataset.get_train(dataset_fold, **data_kwargs)
        X_val, Y_val = dataset.get_val(dataset_fold, **data_kwargs)
        
        train_data = [X_train, Y_train]
        val_data = [X_val, Y_val]
    
    validation_steps = None
    if skips_val:
        print("WARNING: 'Skipping' validation")
        # val_data = None
        validation_steps = 2
    
    num_objs = len(X_train)
    # if not temp_cnn:
    #     object_shape = (len(X_train[0][0]),)
    # else:
    #     _, seq_len, _, *object_shape = np.array(X_train).shape
    #     object_shape = tuple(object_shape)
    
    # timesteps = data_kwargs.get('timesteps', 10)
    if not indiv_out:
        output_size = len(Y_train[0])
        indiv_output_size = 0
    else:
        output_size = len(Y_train[0][0])
        indiv_output_size = len(Y_train[1][0])
    
    group_indivs = data_kwargs.get('group_indivs', False)
    
    ### TODO train, val and test have different cw 
    # (for loss go with train, but for acc it should be specific?)
    if not indiv_out:
        class_weight = get_class_weight(dataset, 
                   indiv_actions=(not group_indivs), dataset_fold=dataset_fold)
        metric_cw = get_class_weight(dataset, 
                   indiv_actions=(not group_indivs), dataset_fold=dataset_fold)
    else:
        class_weight = None
        metric_cw = get_class_weight(dataset, indiv_actions=True, 
                                     dataset_fold=dataset_fold)
    
    ### Hard-coded setting higher value for class_weight
    if not group_indivs and dataset_name == 'Volleyball':
        # For Volleyball dataset
        print("WARNING: Setting higher class_weight to class 'Standing'.")
        multiplier = 2 # 2 5 10
        print("> Multiplier:", multiplier)
        class_weight[8] = class_weight[8]*multiplier
        # print(class_weight[8], multiplier)
    
    if not use_cw:
        # print("WARNING: Setting class_weight to None.")
        class_weight = None
    
    #### Creating model
    if verbose > 0:
        print("Creating model...")
    
    if not temp_cnn:
        if not fusion_cnn:
            model = get_model(num_objs=num_objs, 
                output_size=output_size, indiv_output_size=indiv_output_size,
                drop_rate=drop_rate, **model_kwargs)
        else:
            raise NotImplementedError("fusion_cnn = True")
            # model = get_fused_model(selected_joints, object_shape=object_shape, 
            #     output_size=output_size, indiv_output_size=indiv_output_size,
            #     num_dim=num_dim, overhead=overhead, 
            #     kernel_init_type=kernel_init_type, drop_rate=drop_rate,
            #     kernel_init_param=kernel_init_param, 
            #     kernel_init_seed=kernel_init_seed, 
            #     models_kwargs=model_kwargs, **fusion_kwargs)
    else:
        raise NotImplementedError("temp_cnn = True")
        # print("Warning: Not tested for a while")
        # model = tirn.get_model(num_objs=num_joints, object_shape=object_shape, 
        #     output_size=output_size, num_dim=num_dim, overhead=overhead,
        #     kernel_init_type=kernel_init_type, drop_rate=drop_rate, 
        #     kernel_init_param=kernel_init_param, seq_len=seq_len,
        #     kernel_init_seed=kernel_init_seed, **model_kwargs)
    
    if len(model.input) != len(X_train):
        print("Warning: Model input size differs from data size.")
        print("\t Model:", len(model.input), "Data:", len(X_train))
    
    ### Call train_model()
    fit_history = train_model(model=model, verbose=verbose, 
      learning_rate=learning_rate, output_path=output_path, 
      checkpoint_period=checkpoint_period, batch_size=batch_size,
      epochs=epochs, use_data_gen=use_data_gen, train_data=train_data, 
      val_data=val_data, subsample_ratio=subsample_ratio, 
      use_earlystopping=use_earlystopping, class_weight=class_weight,
      metric_class_weight=metric_cw, grp_out=grp_out,
      indiv_out=indiv_out, indiv_loss_mult=indiv_loss_mult,
      use_lr_sched=use_lr_sched, eval_metric=eval_metric, 
      epoch_verbose=epoch_verbose, validation_steps=validation_steps)
    
    #### HINT Evaluating on test
    if evaluate_on_test:
        if verbose > 0:
            print("Evaluating model on test split...")
            
        if not temp_cnn:
            test_generator = DataGenerator(dataset_name, dataset_fold, 'test',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
                indiv_out=indiv_out, grp_out=grp_out, **data_kwargs)
        else:
            raise NotImplementedError("temp_cnn = True")
            # test_generator = DataGeneratorSeq(dataset_name, dataset_fold,'test',
            #     batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            #     pad_sequences=True, buffer_data=False, **data_kwargs)
    
        ## Run test eval based on lowest val_loss
        
        # If earlystop, it is the best model based on loss, else load weigths
        num_epochs = len(fit_history.history['loss'])
        if num_epochs == epochs: # No early stop, load weights required
            chkpoint_filepath = os.path.join(output_path,'relnet_weights.hdf5')
            model.load_weights(chkpoint_filepath)
        
        Y_pred, Y_true = predict_model(model, test_generator, 
                                   use_data_gen=use_data_gen, verbose=verbose)
        
        ### TODO grp_out==False
        if indiv_out:
            Y_pred, *Y_pred_inds = Y_pred
            Y_true, *Y_true_inds = Y_true
        
        # Make this functions, so it will be easier to do it twice?
        # Sharing function ith predict_irn
        if eval_metric =='accuracy':
            if group_indivs:
                test_gt = None
            else:
                test_gt = dataset.get_split_gt('test', dataset_fold=dataset_fold)
            test_acc = compute_accuracy(Y_true, Y_pred, ref_gt=test_gt)
            fit_history.history['test_accuracy'] = num_epochs*[test_acc]
            fit_history.history['test_acc_by_loss'] = num_epochs*[test_acc]
            
            if indiv_out:
                ### TODO compute acc for indivs
                pass
            
            if verbose > 0:
                print("(Best loss) Test - ACC: {:.2%}".format(test_acc))
        elif eval_metric == 'precision':
            test_prec = compute_precision(Y_true, Y_pred)
            fit_history.history['test_mAP'] = num_epochs*[test_prec]
            
            if verbose > 0:
                print("(Best loss) Test - mAP: {:.3f}".format(test_prec))
                
        ## Now running for best eval metric
        filename = ('relnet_weights-val_acc.hdf5' if eval_metric == 'accuracy'
                    else 'relnet_weights-val_prec.hdf5')
        chkpoint_filepath = os.path.join(output_path, filename)
        model.load_weights(chkpoint_filepath)
        
        Y_pred, Y_true = predict_model(model, test_generator, 
                                   use_data_gen=use_data_gen, verbose=verbose)
        
        if save_preds:
            # save_predictions(Y_pred, Y_true, output_path, indiv_out=indiv_out)
            # if verbose > 0:
            #     print("Predictions and true classes saved.")
            # save_scores(Y_pred, output_path, indiv_out=indiv_out)
            suffix = ''
            save_scores(Y_pred, output_path, indiv_out=indiv_out, suffix=suffix)
            if verbose > 0:
                print("Predictions scores saved.")
        
        if indiv_out:
            Y_pred, *Y_pred_inds = Y_pred
            Y_true, *Y_true_inds = Y_true
        
        if eval_metric == 'accuracy':
            if group_indivs:
                test_gt = None
            else:
                test_gt = dataset.get_split_gt('test', dataset_fold=dataset_fold)
            test_acc2 = compute_accuracy(Y_true, Y_pred, ref_gt=test_gt)
            if test_acc2 > test_acc: # If better, replace
                fit_history.history['test_accuracy'] = num_epochs*[test_acc2]
            fit_history.history['test_acc_by_eval'] = num_epochs*[test_acc2]
            
            if indiv_out:
                ### TODO compute acc for indivs
                pass
            
            if verbose > 0:
                print("(Best eval) Test - ACC: {:.2%}".format(test_acc2))
        elif eval_metric == 'precision':
            test_prec = compute_precision(Y_true, Y_pred)
            fit_history.history['test_mAP'] = num_epochs*[test_prec]
            
            if verbose > 0:
                print("(Best eval) Test - mAP: {:.3f}".format(test_prec))
        
        # Save new train.log with fit_history that includes test results
        log_filepath = os.path.join(output_path, 'training.log')
        if os.path.exists(log_filepath):
            os.rename(log_filepath, log_filepath+'.bkp')
            
            header = ','.join(['epoch'] + list(fit_history.history.keys()))
            epochs = list(range(len(fit_history.history['loss'])))
            values = np.transpose([epochs]+list(fit_history.history.values()))
            np.savetxt(log_filepath, values, fmt='%g', delimiter=',',
                       header=header, comments='')
        
        del test_generator
    
    # Doing some cleaning, attempt to reduce mem error when running in sequence
    del model, train_generator, val_generator
    tf.keras.backend.clear_session()
    
    return fit_history

# %% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting Train CNN -',time.asctime(time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    train_cnn(**args)

    print('\n> Finished Train CMM -',time.asctime(time.localtime(time.time())))
