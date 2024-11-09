import os
import numpy as np

from tensorflow.keras.utils import Sequence

from datasets import Volleyball
from misc.data_io import get_persons_info, get_Y_data, argsort_players
from misc.video_data_io import get_data
from misc.utils import get_class_weight

class DataGenerator(Sequence):
    def __init__(self, dataset_name, dataset_fold, subset,
                batch_size=48, reshuffle=False, shuffle_indiv_order=False, 
                shuffle_sides=False, indiv_out=False,
                grp_out=None, 
                jitter_timesteps=False, jitter_timesteps_mag=1,
                ind_clip_weights=None, use_cw=True, **data_kwargs):
        self.requires_roi = False
        self.requires_bboxes = False
        if dataset_name == 'Volleyball':
            dataset = Volleyball
            # self.pose_style = 'OpenPose-preproc'
            data_kwargs.setdefault('balls_hdf_filepath',
                               os.path.join(Volleyball.DATA_DIR, 'balls.hdf5'))
            self.requires_bboxes = True
        
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.subset = subset
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.shuffle_indiv_order = shuffle_indiv_order
        self.shuffle_sides = shuffle_sides
        self.data_kwargs = data_kwargs
        self.group_indivs = data_kwargs.get('group_indivs', False)
        self.indiv_out = indiv_out
        self.grp_out = (grp_out if grp_out is not None else self.group_indivs)
        self.use_cw = use_cw
        self.ind_clip_weights = ind_clip_weights
        self.jitter_timesteps = jitter_timesteps
        self.jitter_timesteps_mag = jitter_timesteps_mag
        
        self.ground_truth = dataset.get_split_gt(split=subset,
                                                  dataset_fold=dataset_fold)
        
        if self.group_indivs:
            self.num_classes_grp = len(dataset.GRP_ACTIVITIES)
            if not indiv_out:
                self.num_classes_indiv = None
            else:
                self.num_classes_indiv = len(dataset.IND_ACTIONS)
        else:
            self.num_classes_indiv = len(dataset.IND_ACTIONS)
            self.num_classes_grp = None
            
        persons_info = get_persons_info(self.ground_truth.iloc[0], False)
        self.num_inds = len(persons_info)
        
        if self.use_cw:
            self.grp_cw = get_class_weight(self.dataset, indiv_actions=False, 
                                  split=self.subset, dataset_fold=dataset_fold)
        
            self.ind_cw = get_class_weight(self.dataset, indiv_actions=True, 
                                  split=self.subset, dataset_fold=dataset_fold)
            
            if self.ind_clip_weights is not None:
                w_min, w_max = self.ind_clip_weights
                for k, w in self.ind_cw.items():
                    self.ind_cw[k] = np.clip(w, w_min, w_max)
        
        if batch_size%self.num_inds != 0 and not self.group_indivs:
            print("Batch size should be multiple of the number of persons:"
                  , self.num_inds)
            self.batch_size -= batch_size % self.num_inds
            print("Changed value to:", self.batch_size)
        
        num_seqs = self.ground_truth.shape[0]
        if subset == 'train':
            self.shuffled_idx = np.random.choice(self.ground_truth.index.values, 
                num_seqs, replace=False)
        elif subset == 'validation' or subset == 'test':
            self.shuffled_idx = self.ground_truth.index.values
        
        if self.group_indivs:
            self.num_batches = int(np.ceil(num_seqs/batch_size))
        else:
            self.num_batches = int(np.ceil(self.num_inds*num_seqs/batch_size))
        
        # Validating data_kwargs
        # print("WARNING: not validating get data!!!!!")
        sample_gt = self.ground_truth.sample()
        if not self.requires_roi and not self.requires_bboxes:
            _ = get_data(sample_gt, 
                num_classes_indiv=self.num_classes_indiv, 
                num_classes_grp=self.num_classes_grp, 
                **self.data_kwargs)
        else:
            if self.requires_roi and 'event_rois' not in sample_gt:
                sample_gt = self.dataset.add_mult_event_rois(sample_gt)
            if self.requires_bboxes and 'event_bboxes' not in sample_gt:
                sample_gt = self.dataset.add_mult_event_bboxes(sample_gt)
            _ = get_data(sample_gt, 
                num_classes_indiv=self.num_classes_indiv, 
                num_classes_grp=self.num_classes_grp, 
                **self.data_kwargs)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        if self.group_indivs:
            batch_slice = slice(self.batch_size*idx, self.batch_size*(idx+1))
        else:
            batch_slice = slice(self.batch_size*idx//self.num_inds,
                                self.batch_size*(idx+1)//self.num_inds)
        batch_idxs = self.shuffled_idx[batch_slice]
        
        batch_gt = self.ground_truth.loc[batch_idxs]
        ## Populate batch_gt with required extra info
        if self.requires_roi and 'event_rois' not in batch_gt:
            batch_gt = self.dataset.add_mult_event_rois(batch_gt)
        if self.requires_bboxes and 'event_bboxes' not in batch_gt:
            batch_gt = self.dataset.add_mult_event_bboxes(batch_gt)
        
        batch_x, batch_y = get_data(batch_gt, 
             num_classes_indiv=self.num_classes_indiv, 
             num_classes_grp=self.num_classes_grp, **self.data_kwargs)
        
        ## Data Augmentation checks
        
        if self.shuffle_indiv_order:
            # This shuffle is only useful when pool with concat
            raise NotImplementedError("shuffle_indiv_order = True")
        
        if self.jitter_timesteps:
            raise NotImplementedError("jitter_timesteps = True")
        
        if self.shuffle_sides:
            batch_x = np.asarray(batch_x)
            # Randomly select half
            rnd_ids = np.random.choice(range(batch_x.shape[0]), 
                                       size=batch_x.shape[0]//2, replace=False)
            # Randomly select
            # rnd_ids = np.random.randint(2, size=batch_x.shape[0], dtype=bool)
            
            batch_x[rnd_ids] = np.flip(batch_x[rnd_ids], -2)
            if self.data_kwargs['data_type'] == 'flows':
                ## Getting negative of horizontal component
                # batch_x[rnd_ids,...,::2] = 255 - batch_x[rnd_ids,...,::2]
                negative_imgs = 255 - batch_x[rnd_ids,...,::2]
                negative_imgs[negative_imgs==255] = 0 # Keeping borders black
                batch_x[rnd_ids,...,::2] = negative_imgs
            
            # Change group activity labels sides, if present
            if self.group_indivs and self.dataset_name == 'Volleyball':
                # Considering indexes of "opposite" classes are mirrored
                # batch_y[rnd_ids] = self.num_classes_grp - batch_y - 1
                if not self.indiv_out:
                    batch_y[rnd_ids] = np.flip(batch_y[rnd_ids], axis=-1)
                else:
                    batch_y[0][rnd_ids] = np.flip(batch_y[0][rnd_ids], axis=-1)
        
        # Squeeze if single frame
        if len(batch_x.shape) == 6 and batch_x.shape[2] == 1:
            batch_x = batch_x.squeeze(2)
        
        # batch_x = [ np.asarray(sample_x) for sample_x in batch_x]
        if self.group_indivs:
            # batch_x = [np.asarray(batch_x[:,i]) for i in range(batch_x.shape[1])]
            # if len(batch_x.shape) == 6 and batch_x.shape[0] == 1:
            if len(batch_x.shape) == 6 and self.batch_size == 1:
                # Squeeze if single video and it was not squeezed before
                # This is used for frames temp pool pred
                batch_x = [np.asarray(batch_x[0,i]) 
                           for i in range(batch_x.shape[1])]
            else:
                batch_x = [np.asarray(batch_x[:,i]) 
                           for i in range(batch_x.shape[1])]
        
        
        ret = [batch_x, batch_y]
        
        if self.indiv_out:
            sample_weights = self.get_sample_weight(batch_y, batch_gt)
            ret.append(sample_weights)
        
        if self.group_indivs and not self.grp_out:
            assert self.indiv_out # At least indiv out must be true
            ret = [batch_x, batch_y[1:], sample_weights[1:]]
        
        return tuple(ret)
    
    def on_epoch_end(self):
        if self.reshuffle:
            self.shuffled_idx = np.random.choice(
                self.ground_truth.index.values, 
                self.ground_truth.shape[0], replace=False)
    
    def get_y(self, idx):
        if self.group_indivs:
            batch_slice = slice(self.batch_size*idx, self.batch_size*(idx+1))
        else:
            batch_slice = slice(self.batch_size*idx//self.num_inds,
                                self.batch_size*(idx+1)//self.num_inds)
        batch_idxs = self.shuffled_idx[batch_slice]
        batch_gt = self.ground_truth.loc[batch_idxs]
        
        batch_y = get_Y_data(batch_gt,num_classes_indiv=self.num_classes_indiv, 
                 num_classes_grp=self.num_classes_grp, 
                 num_indivs=self.data_kwargs.get('num_indivs', self.num_inds),
                 indiv_ova_class=self.data_kwargs.get('indiv_ova_class'),
                 sort_players=self.data_kwargs.get('sort_players', False),
                 )
        
        if self.group_indivs and not self.grp_out:
            assert self.indiv_out # At least indiv out must be true
            batch_y = batch_y[1:]
        
        return batch_y
    
    def get_sample_weight(self, batch_y, batch_gt):
        # Assuming is always grp+indivs
        Y_grp, *Y_indivs  = batch_y
        
        if self.use_cw:
            ## Setting grp_sw as ones, i.e. use cw only for indivs.
            # grp_sw = [self.grp_cw[cls_idx] for cls_idx in Y_grp.argmax(axis=-1)]
            grp_sw = [1] * len(Y_grp)
        
            inds_sw = [ [ self.ind_cw[cls_idx] 
                         for cls_idx in Y_indiv.argmax(axis=-1) ] 
                       for Y_indiv in Y_indivs ]
        else:
            # grp_sw = [ 1 for cls_idx in Y_grp.argmax(axis=-1) ]
            # inds_sw = [ [ 1 for cls_idx in Y_indiv.argmax(axis=-1) ] 
            #            for Y_indiv in Y_indivs ]
            grp_sw = [1] * len(Y_grp)
            inds_sw = np.ones(np.shape(Y_indivs)[:-1], dtype=int).tolist()
        
        for video_idx in range(len(batch_gt)):
            video_gt = batch_gt.iloc[video_idx]
            
            if not self.data_kwargs.get('sort_players', False):
                persons_info = get_persons_info(video_gt, prune_missing=False)
                num_indivs = self.data_kwargs.get('num_indivs', len(persons_info))
                persons_info = persons_info.head(num_indivs)
                missing_persons_mask = persons_info.action.str.match('missing')
                padded_persons = persons_info[missing_persons_mask].index
            else:
                sorted_idx = np.asarray(argsort_players(video_gt))
                padded_persons = np.argwhere(sorted_idx == None).flatten()
            
            for padded_idx in padded_persons:
                inds_sw[padded_idx][video_idx] = 0
        
        sample_weights = [grp_sw] + inds_sw
        sample_weights = [ np.asarray(w) for w in sample_weights ]
        
        return tuple(sample_weights)
        
