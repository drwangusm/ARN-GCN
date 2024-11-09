import configparser, os, glob, h5py
from ast import literal_eval
import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

def parse_logs(logs):
    is_dataframe = isinstance(logs, pd.DataFrame)
    if not is_dataframe:
        logs = logs.copy()
    else:
        logs = logs.to_dict()
    
    if 'softmax_accuracy_group' in logs:
        logs['accuracy'] = logs.pop('softmax_accuracy_group')
        logs['val_accuracy'] = logs.pop('val_softmax_accuracy_group')
        
        logs['grp_loss'] = logs.pop('softmax_loss')
        logs['val_grp_loss'] = logs.pop('val_softmax_loss')
    
    acc_ind_keys = [ k for k in logs if 'accuracy_indiv' in k ]
    if acc_ind_keys != []:
        acc_ind_key = acc_ind_keys[0]
        if 'accuracy' in logs:
            logs['accuracy_indiv'] = logs.pop(acc_ind_key)
            logs['val_accuracy_indiv'] = logs.pop('val_'+acc_ind_key)
        else:
            logs['accuracy'] = logs.pop(acc_ind_key)
            logs['val_accuracy'] = logs.pop('val_'+acc_ind_key)
            
        
    mpca_ind_keys = [ k for k in logs if 'mpca_indiv' in k ]
    if mpca_ind_keys != []:
        mpca_ind_key = mpca_ind_keys[0]
        logs['mpca_indiv'] = logs.pop(mpca_ind_key)
        logs['val_mpca_indiv'] = logs.pop('val_'+mpca_ind_key)
    
    if is_dataframe:
        logs = pd.DataFrame.from_dict(logs)
    
    return logs

def find_best_weights(base_path, criteria='val_loss', verbose=0):
    from misc.print_train_stats import pretty_print_stats
    rerun_paths = glob.glob(base_path+'/rerun_*/')
    rerun_paths += glob.glob(base_path+'/fold_*/')
    rerun_paths.sort()
    
    best_epochs = []
    for rerun_path in rerun_paths:
        rerun_df = pd.read_csv(rerun_path + 'fit_history.csv')
        rerun_df['path'] = rerun_path
        
        if criteria.endswith('loss'):
            best_epoch = rerun_df.loc[rerun_df[criteria].idxmin()]
        # elif criteria.endswith('acc'):
        elif 'acc' in criteria:
            sorted_rerun_df = rerun_df.sort_values([criteria, 'val_loss'], 
                ascending=[False, True])
            best_epoch = sorted_rerun_df.iloc[0]
        
        best_epochs.append(best_epoch)
    
    summary_df = pd.concat(best_epochs, axis=1).T.reset_index(drop=True)
    summary_df = summary_df.astype({criteria: 'float'})
    
    if criteria.endswith('loss'):
        best_rerun = summary_df.loc[summary_df[criteria].idxmin()]
        weights_path = best_rerun.path + 'relnet_weights.hdf5'
    elif criteria.endswith('acc'):
        # TODO replace this with 'acc' in 
        sorted_summary_df = summary_df.sort_values([criteria, 'val_loss'], 
            ascending=[False, True])
        best_rerun = sorted_summary_df.iloc[0]
        weights_path = best_rerun.path + 'relnet_weights-val_acc.hdf5'
    
    if verbose > 0:
        print("Best weights stats:")
        best_rerun_df = best_rerun.to_frame().transpose()
        best_rerun_df = best_rerun_df.astype(
            {'acc': 'float', 'loss': 'float', 'val_acc': 'float', 'val_loss': 'float'})
        pretty_print_stats(best_rerun_df)

    return weights_path

def read_config(config_filepath, fusion=False):
    def unstringify_dict(d):
        # literal_eval accepts only: strings, numbers, tuples, lists, dicts, 
        #  booleans, and None
        # return dict((k,literal_eval(v)) for k,v in d.items())
        d_ret = {}
        for k,v in d.items():
            if not v.startswith('slice'):
                d_ret[k] = literal_eval(v)
            else:
                tup = literal_eval(v[5:])
                d_ret[k] = slice(*tup)
        return d_ret
    
    config = configparser.ConfigParser()
    with open(config_filepath) as config_file:
        config.read_file(config_file)
    
    if not fusion:
        data_kwargs = unstringify_dict(config['data'])
        model_kwargs = unstringify_dict(config['model'])
        kwargs = [data_kwargs, model_kwargs]
    else:
        fusion_kwargs = unstringify_dict(config['fusion'])
        kwargs = [fusion_kwargs]
    train_kwargs = unstringify_dict(config['train'])
    
    kwargs.append(train_kwargs)
    
    return tuple(kwargs)

def save_config(config_filepath, data_kwargs, model_kwargs, train_kwargs,
        fusion_kwargs = None):
    def stringify_dict(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, str):
                new_d[k] = "'"+v+"'"
            else:
                new_d[k] = str(v)
        return new_d
        # return dict((k,str(v)) for k,v in d.items())

    config = configparser.ConfigParser()
    
    if fusion_kwargs is None:
        config['data'] = stringify_dict(data_kwargs)
        config['model'] = stringify_dict(model_kwargs)
    else:
        config['fusion'] = stringify_dict(fusion_kwargs)
    config['train'] = stringify_dict(train_kwargs)
    
    print("Saving configuration at:", config_filepath)
    with open(config_filepath, 'w') as config_file:
        config.write(config_file)

def get_fusion_kwargs(config_filepaths):
    models_kwargs = []
    base_data_kwargs = {}
    all_selected_joints = set()
    for config_filepath in config_filepaths:
        data_kwargs, model_kwargs, _ = read_config(config_filepath)
        
        base_data_kwargs.update(data_kwargs)
        selected_joints = data_kwargs.get('selected_joints', [])
        model_kwargs['selected_joints'] = selected_joints
        models_kwargs.append(model_kwargs)
        # all_selected_joints = all_selected_joints.union(set(selected_joints))
        all_selected_joints.update(set(selected_joints))
    
    base_data_kwargs['selected_joints'] = list(all_selected_joints)
    
    return base_data_kwargs, models_kwargs

def get_persons_info(video_gt, prune_missing=True):
    persons_info = []
    
    p_labels = [ col[:-2] for col in video_gt.index if col.endswith('_x')]
    for p_label in p_labels:
        p_cols = [ col for col in video_gt.index 
                  if col.startswith(p_label+'_')]
        p_info = list(video_gt[p_cols].values)
        persons_info.append(p_info)
    # persons_info_fields = ['x','y','width','height','action','action_id']
    persons_info_fields = [ col[3:] for col in video_gt.index if col.startswith('p1_')]
    persons_info = pd.DataFrame(persons_info, columns=persons_info_fields)
    if prune_missing:
        persons_info = persons_info[~persons_info.action.str.match('missing')]
    return persons_info

def compute_class_weight(classes, y):
    n_samples = len(y)
    n_classes = len(classes)
    
    class_weight = n_samples / (n_classes * np.bincount(y))
    
    class_weight = dict(enumerate(class_weight))
    
    return class_weight

def get_class_weight(dataset, indiv_actions=False, split='train', dataset_fold=None):
    if indiv_actions:
        sufix = '_action_id'
    else:
        sufix = 'grp_activity_id'
    
    split_gt = dataset.get_split_gt(split, dataset_fold)
    col_names = [ col for col in split_gt.columns if col.endswith(sufix)]
    label_idxs = split_gt[col_names].values
    most_common = np.bincount(label_idxs[label_idxs!=-1]).argmax()
    label_idxs[label_idxs==-1] = most_common # Converting missing
    classes = np.unique(label_idxs) # has -1
    y = label_idxs.flat
    
    class_weight = compute_class_weight(classes, y)
    
    return class_weight

def display_frames(frames, title='', waitKey=0, add_frame_num=True):
    import cv2
    speed = 1.
    frame_idx = 0
    key = -1
    
    while key != ord('q'):
    # while frame_idx < len(frames):
        # cv2.imshow(title, frames[frame_idx])
        numbered_frame = frames[frame_idx].copy()
        
        if len(numbered_frame.shape) == 2:
            numbered_frame = cv2.cvtColor(numbered_frame, cv2.COLOR_GRAY2BGR)
        
        if add_frame_num:
            cv2.putText(numbered_frame, str(frame_idx),
                org=(10,25), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=.75,
                color=(0,0,255), # Red
                thickness=2,
                lineType=cv2.LINE_AA)
        cv2.imshow(title, numbered_frame)
        key = cv2.waitKeyEx(int(waitKey*speed))
        if key == ord('q'):
            break
        elif key == 2424832: # Left arrow
            frame_idx = max(frame_idx-1, 0)
        elif key == 2555904: # Right arrow
            frame_idx = min(frame_idx+1, len(frames) - 1)
        elif key == ord('f'): # first
            frame_idx = 0
        elif key == ord('l'): # last
            frame_idx = len(frames) - 1
        elif key == ord('m') or key == ord('c'): # middle / center
            frame_idx = len(frames)//2
        elif key == ord('+'): # speed up
            speed -= .25
            speed = max(.25, speed)
        elif key == ord('-'): # slow down
            speed += .25
            speed = min(3, speed)
        elif key == ord('s'): # stop auto change
            last_speed = speed
            speed = 0
        elif key == ord(' '): # pause/resume
            if speed > 0:
                last_speed = speed
                speed = 0
            else:
                speed = last_speed
        elif key == -1: # No key pressed, go to next and loop
            frame_idx += 1
            if frame_idx == len(frames):
                frame_idx = 0
        else: # next
            frame_idx += 1
            frame_idx = min(frame_idx, len(frames) - 1)
    cv2.destroyAllWindows()

def read_video(video_filepath, temp_window=20):
    import cv2
    frames = []
    
    if os.path.isfile(video_filepath):
        if video_filepath.endswith(('jpg','png')):
            dirpath = os.path.dirname(video_filepath)
            filename = os.path.basename(video_filepath)
            mid_frame_num = int(filename.replace('frame','')[:-4])
            ext = filename[-4:]
            # file_tpl = ('frame{:04d}' if 'frame' in filename else '{}') + ext
            
            if 'frame' in filename:
                len_idx = len(filename.replace('frame','')[:-4])
                file_tpl = 'frame{:0{}d}' + ext
            else:
                len_idx = 0
                file_tpl = '{}' + ext
            
            ## In case temp_window is even, the bigger part is before mid
            range_start = mid_frame_num - int(np.ceil((temp_window-1)/2))
            range_end = 1 + mid_frame_num + int(np.floor((temp_window-1)/2))
            for frame_num in range(range_start, range_end):
                frame_filepath = os.path.join(dirpath, 
                                          file_tpl.format(frame_num, len_idx))
                if os.path.exists(frame_filepath):
                    frame = cv2.imread(frame_filepath)
                else:
                    raise IOError("Unable to read frame: "+frame_filepath)
                    # frame = np.zeros_like(frames[-1])
                frames.append(frame)
        else: # Assume it is a video file
            cap = cv2.VideoCapture(video_filepath)
            if not cap.isOpened():
                raise IOError("Unable to read video: "+video_filepath)
            ret, frame = cap.read()
            while ret:
                frames.append(frame)
                ret, frame = cap.read()
            cap.release()
    else:
        # frames_filepaths = sorted(glob.glob(video_filepath+'/*jpg'))
        frames_filepaths = glob.glob(video_filepath+'/*jpg')    
        frames_filepaths += glob.glob(video_filepath+'/*png')  
        
        if 'frame' not in os.path.basename(frames_filepaths[0]):
            frames_filepaths.sort(key=lambda p: int(os.path.basename(p)[:-4]))
        else:
            frames_filepaths.sort()
        
        if frames_filepaths == []:
            raise IOError("Unable to read frames from: "+video_filepath)
        for frame_filepath in frames_filepaths:
            frame = cv2.imread(frame_filepath)
            frames.append(frame)
    
    return frames

def read_flows(flows_dirpath, direction, temp_window=None, mid_frame_num=None):
    import cv2
    flows = []
    
    if temp_window is None:
        if direction == 'u':
            flows_filepaths = glob.glob(flows_dirpath+'/u_*jpg')    
            flows_filepaths += glob.glob(flows_dirpath+'/u_*png')    
        # elif direction == 'v':
        else:
            flows_filepaths = glob.glob(flows_dirpath+'/v_*jpg')    
            flows_filepaths += glob.glob(flows_dirpath+'/v_*png')   
    else:
        ext = '.jpg'
        file_tpl = direction+'_{}' + ext
        
        flows_filepaths = []
        range_start = mid_frame_num - int(np.ceil((temp_window-1)/2))
        range_end = 1 + mid_frame_num + int(np.floor((temp_window-1)/2))
        for frame_num in range(range_start, range_end):
            flow_filepaths = os.path.join(flows_dirpath, 
                                          file_tpl.format(frame_num))
            if not os.path.isfile(flow_filepaths):
                raise IOError("Unable to read flow: "+flow_filepaths)
            flows_filepaths.append(flow_filepaths)
        
        
    flows_filepaths.sort(key=lambda p: int(os.path.basename(p)[2:-4]))
    if flows_filepaths == []:
        raise IOError("Unable to read flows from: "+flows_dirpath)
    for flow_filepath in flows_filepaths:
        flow = cv2.imread(flow_filepath, cv2.IMREAD_GRAYSCALE)
        flows.append(flow)
    
    return flows

def save_frames(output_folder, frames, reps=None):
    import cv2
    for idx, frame in enumerate(frames):
        if reps is None or reps == 0:
            output_filepath = os.path.join(output_folder, 
                                           '{:03d}.png'.format(idx))
            success = cv2.imwrite(output_filepath, frame)
        else:
            base_idx = idx*reps
            for rep_idx in range(reps):
                output_filepath = os.path.join(output_folder, 
                                       '{:03d}.png'.format(base_idx+rep_idx))
                success = cv2.imwrite(output_filepath, frame)
                
        if not success:
            print("Error while saving frame. Stopping.")
            print(idx, frame.shape)
            print(output_filepath)
            break

def get_quadrants(width, height, size):
    
    w_step = width//size
    rem_w = width%size
    h_step = height//size
    rem_h = height%size
    quadrants_slices = []
    for i in range(size**2):
        w_i = i%size
        rem = (0 if w_i != size-1 else rem_w)
        w_slice = slice(w_i*w_step, (w_i+1)*w_step + rem)
        
        h_i = i//size
        rem = (0 if h_i != size-1 else rem_h)
        h_slice = slice(h_i*h_step, (h_i+1)*h_step + rem)
        quadrants_slices.append([h_slice, w_slice])
    return quadrants_slices

def estimate_motion_per_quadrant(frames, size=3):
    frames = np.asarray(frames)
    h, w, _ = frames[0].shape
    
    quadrants_slices = get_quadrants(w, h, size)
    
    quadrant_motions = []
    for quadrant in quadrants_slices:
        frames_quadrant = frames[:,quadrant[0],quadrant[1]]
        quadrant_motions.append( estimate_motion(frames_quadrant) )
    quadrant_motions = np.asarray(quadrant_motions)
    
    return quadrant_motions

def create_motion_map(quadrant_motions, width, height):
    import cv2
    
    quads_size = np.sqrt(len(quadrant_motions)).astype(int)
    quadrants_slices = get_quadrants(width, height, quads_size)
    
    quad_means_motions = quadrant_motions.mean(axis=0)
    
    num_motions = quadrant_motions.shape[1]
    motion_maps = np.zeros((num_motions, height, width, 2), dtype=np.float32)
    for motion_idx in range(num_motions):
        mean_motion = quad_means_motions[motion_idx]
        for quad_idx, quadrant in enumerate(quadrants_slices):
            quad_frm_motions = quadrant_motions[quad_idx, motion_idx]
            quad_frm_motions = np.average([quad_frm_motions, mean_motion], 0)
            motion_maps[motion_idx,quadrant[0],quadrant[1]] = quad_frm_motions
        
        motion_maps[motion_idx] = cv2.boxFilter(motion_maps[motion_idx], 
                                                ddepth=-1, ksize=(25,25))
    
    return motion_maps

def find_quadrant_idx(x, y, width, height, size):
    w_step = width//size
    h_step = height//size
    
    w_i = x//w_step
    h_i = y//h_step
    
    quad_idx = w_i + h_i*size
    
    return quad_idx
    
def apply_cam_motion_per_quad(video_data, cam_motions, width, height,
              input_type='coords', round_int=False, poses=False, 
              use_motion_map=True, undo=False):
    # NOTE: Undo is not realiable, there is significant noise included
    if poses:
        video_data = [ [ pose['coords'] for pose in frame_poses ] 
                         for frame_poses in video_data ]
    
    quads_size = np.sqrt(len(cam_motions)).astype(int)
    
    if use_motion_map:
        motion_maps = create_motion_map(cam_motions, width, height)
    
    new_video_data = [video_data[0].copy()]
    for frame_idx in range(1, len(video_data)):
    # for frame_idx in range(1, 10):
        # print("> frame_idx", frame_idx)
        frame_data = video_data[frame_idx]
        new_frame_data = []
        for data in frame_data:
            new_data = np.array(data)
            # print("> > new_data:", new_data)
            
            motions_iter = reversed(range(frame_idx))
            if undo:
                motions_iter = range(frame_idx)
            
            for motion_idx in motions_iter:
                if input_type == 'coords':
                    x, y = new_data[0]
                elif input_type == 'bboxes':
                    x, y = new_data[:2]
                
                if not use_motion_map:
                    quad_idx = find_quadrant_idx(x, y, width, height, quads_size)
                    curr_cum_motion = cam_motions[quad_idx, motion_idx]
                    # print(x, y, quad_idx, motion_idx)
                else:
                    curr_cum_motion = motion_maps[motion_idx, y, x]
                
                if undo:
                    curr_cum_motion = -curr_cum_motion
                
                if input_type == 'coords':
                    non_zero = np.all(new_data, axis=1)
                    new_data[non_zero] = new_data[non_zero] - curr_cum_motion
                elif input_type == 'bboxes':
                    new_x = new_data[0] - curr_cum_motion[0]
                    new_y = new_data[1] - curr_cum_motion[1]
                    
                    if round_int:
                        new_x = int(round(new_x))
                        new_y = int(round(new_y))
                    
                    # new_data = [new_x, new_y] + list(data[2:])
                    new_data[:2] = [new_x, new_y]
            
            new_frame_data.append(new_data)
        new_video_data.append(new_frame_data)
    
    if poses:
        new_video_data = [ [ {'coords': data} for data in frame_data ] 
                         for frame_data in new_video_data ]
    
    return new_video_data

def estimate_motion(frames, smooth=True, clean_outliers=False):
    import cv2
    n_frames = len(frames)
    
    prev = frames[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # Pre-define transformation-store array
    # transforms = np.zeros((n_frames-1, 3), np.float32) 
    transforms = np.zeros((n_frames-1, 2), np.float32) 
    
    for i in range(n_frames-1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)
        
        # Read next frame
        curr = frames[i+1]
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
         
        if prev_pts is None: # No feature detected, skip it
            transforms[i] = [0,0]
            prev_gray = curr_gray
            continue
        
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
        
        # Sanity check
        assert prev_pts.shape == curr_pts.shape 
        
        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if clean_outliers:
            # print("> frame:", i)
            dists = np.linalg.norm(curr_pts - prev_pts, axis=2).squeeze()
            # print(dists.min(), dists.mean(), dists.max(), 2*dists.std())
            clean_idx = (np.abs(dists - dists.mean()) < 2*dists.std())
            prev_pts = prev_pts[clean_idx]
            curr_pts = curr_pts[clean_idx]
            # print(dists.shape[0])
            # print(clean_idx.sum())
        
        
        #Find transformation matrix
        if prev_pts.size > 0:
            m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)
            # will only work with OpenCV-3 or less
        else:
            m = None
        
        # Extract traslation
        if m is not None:
            dx = m[0,2]
            dy = m[1,2]
        else:
            # dx = dy = 0
            dx, dy = transforms[max(0, i-1)]
        
        # print([dx,dy])
        
        # Extract rotation angle
        # da = np.arctan2(m[1,0], m[0,0])
         
        # Store transformation
        # transforms[i] = [dx,dy,da]
        transforms[i] = [dx,dy]
         
        # Move to next frame
        prev_gray = curr_gray
    
    if smooth:
        transforms = uniform_filter1d(transforms, size=3, axis=0)
    
    return transforms

def video_poses_to_coords(video_poses):
    video_coords = [ [ pose['coords'] for pose in frame_poses ] 
                    for frame_poses in video_poses ]
    return video_coords

def video_coords_to_poses(video_coords):
    video_poses = [ [ {'coords': coords} for coords in frame_coords ] 
                         for frame_coords in video_coords ]
    return video_poses

def apply_cam_motion(video_data, cam_motions, input_type='coords', 
                     round_int=False, poses=False):
    cum_cam_motions = cam_motions.cumsum(axis=0)
    
    if poses:
        video_data = video_poses_to_coords(video_data)
    
    new_video_data = [video_data[0].copy()]
    # for frame_idx in range(len(video_data)-1):
    # for motion_idx in range(len(cum_cam_motions)):
    # for frame_data in video_data:
    for motion_idx, curr_cum_motion in enumerate(cum_cam_motions):
        frame_data = video_data[motion_idx+1]
        new_frame_data = []
        for data in frame_data:
            if input_type == 'coords':
                new_data = np.array(data)
                non_zero = np.all(new_data, axis=1)
                new_data[non_zero] = new_data[non_zero] - curr_cum_motion
            elif input_type == 'bboxes':
                new_x = data[0] - curr_cum_motion[0]
                new_y = data[1] - curr_cum_motion[1]
                
                if round_int:
                    new_x = int(round(new_x))
                    new_y = int(round(new_y))
                
                new_data = [new_x, new_y] + list(data[2:])
            new_frame_data.append(new_data)
        new_video_data.append(new_frame_data)
    
    if poses:
        new_video_data = video_coords_to_poses(new_video_data)
    
    return new_video_data

def get_bboxes(all_coords, add_label=True, ignore_zeros=True):
    bboxes = []
    all_coords = np.asarray(all_coords)
    for idx, coords in enumerate(all_coords):
        x_coords = coords[:,0]
        y_coords = coords[:,1]
        if np.any(x_coords) and np.any(y_coords):
            x = int(np.min(x_coords[np.nonzero(x_coords)]))
            y = int(np.min(y_coords[np.nonzero(y_coords)]))
            w = int(np.max(x_coords[np.nonzero(x_coords)])) - x
            h = int(np.max(y_coords[np.nonzero(y_coords)])) - y
            if add_label:
                label = str(idx)
                bboxes.append([x,y,w,h,label])
            else:
                bboxes.append([x,y,w,h])
        elif not ignore_zeros:
            if add_label:
                label = str(idx)
                bboxes.append([0,0,1,1,label])
            else:
                bboxes.append([0,0,1,1])
    
    return bboxes

def get_point_bbox(point, size=40):
    if tuple(point) == (0,0):
        bbox = None
    else:
        x, y = point
        bbox = [x-size//2, y-size//2, size, size]
        bbox = [int(v) for v in bbox]
    return bbox

def get_video_center_bbox(video_gt, ratio=0.75, square=False, include_top=False):
    w = int(video_gt.VideoWidth*ratio)
    h = int(video_gt.VideoHeight*ratio)
    x = (video_gt.VideoWidth - w)//2
    y = (video_gt.VideoHeight - h)//2
    center_bbox = [x, y, w, h]
    
    if square:
        center_bbox = transform_bbox(center_bbox, square=True)
    
    if include_top:
        center_bbox[3] = center_bbox[1] + center_bbox[3]
        center_bbox[1] = 0
    
    return center_bbox

def transform_bbox(bbox, ratio=None, abs_values=None, square=False, pad_top=None):
    x, y, w, h = bbox[:4]
    
    if ratio is not None:
        new_w = int(w*ratio)
        new_h = int(h*ratio)
        if ratio > 1:
            new_x = x - (new_w - w)//2
            new_y = y - (new_h - h)//2
        else:
            new_x = x + (w - new_w)//2
            new_y = y + (h - new_h)//2
    
    if abs_values is not None:
        new_w = int(w + abs_values[0])
        if new_w > w:
            new_x = x - (new_w - w)//2
        else:
            new_x = x + (w - new_w)//2
            
        new_h = int(h + abs_values[1])
        if new_h > h:
            new_y = y - (new_h - h)//2
        else:
            new_y = y + (h - new_h)//2
    
    if square:
        if w > h:
            diff = w - h
            new_w = int(w - diff)
            new_x = x + diff//2
            new_y = y
            new_h = h
        else:
            diff = h - w
            new_h = int(h - diff)
            new_y = y + diff//2
            new_x = x
            new_w = w
    
    if pad_top is not None:
        new_x = x
        new_w = w
        new_y = y - pad_top
        new_h = h + pad_top
        
    
    new_bbox = [new_x, new_y, new_w, new_h] + list(bbox[4:])
    return new_bbox

def is_inside(point, bbox):
    x1, y1, w, h = bbox[:4]
    x2 = x1 + w
    y2 = y1 + h
    return x1 < point[0] < x2 and y1 < point[1] < y2

def comp_bboxes_dist(bbox1, bbox2):
    bboxes_pts = []
    for bbox in [bbox1, bbox2]:
        x, y, w, h = bbox[:4]
        bboxes_pts.append([[x,y], [x + w, y + h]])
    bboxes_pts = np.asarray(bboxes_pts)
    # print(bboxes_pts)
    
    dists = []
    for dim_idx in range(2):
        o1_p1, o1_p2 = bboxes_pts[0,:,dim_idx]
        o2_p1, o2_p2 = bboxes_pts[1,:,dim_idx]
        # print(o1_p1 , o2_p1 ,  o1_p2)
        # print(o1_p1 , o2_p2 ,  o1_p2)
        if ((o1_p1 <= o2_p1 <=  o1_p2) or (o1_p1 <= o2_p2 <=  o1_p2) or
                (o2_p1 <= o1_p1 <=  o2_p2)):
            dist_dim = 0
        else:
            if o1_p2 < o2_p1:
                dist_dim = o2_p1 - o1_p2
            else:
                dist_dim = o1_p1 - o2_p2
        dists.append(dist_dim)
    # print(dists)
    
    bboxes_dist = np.linalg.norm(dists)
    return bboxes_dist
    
def merge_bboxes(bboxes, ignore_negs_bbox=True):
    if ignore_negs_bbox:
        # Ignore bboxes with negative locations
        bboxes = [ bbox for bbox in bboxes if (bbox[0] >= 0 and bbox[1] >= 0) ]
    
    # Find smallest x and y
    min_x = np.min([ bbox[0] for bbox in bboxes])
    min_y = np.min([ bbox[1] for bbox in bboxes])
    # Find biggest x+w and y+h
    max_x = np.max([ bbox[0]+bbox[2] for bbox in bboxes])
    max_y = np.max([ bbox[1]+bbox[3] for bbox in bboxes])
    
    merged_bboxes = [min_x, min_y, max_x-min_x, max_y-min_y ]
    return merged_bboxes

def intersection_bboxes(bboxes):
    # Ignore bboxes with negative locations
    # bboxes = [ bbox for bbox in bboxes if (bbox[0] >= 0 and bbox[1] >= 0) ]
    
    if comp_bboxes_dist(bboxes[0], bboxes[1]) > 0:
        # No intersection
        return bboxes[0]
    
    # Find smallest x and y
    x1 = np.max([ bbox[0] for bbox in bboxes])
    y1 = np.max([ bbox[1] for bbox in bboxes])
    # Find biggest x+w and y+h
    x2 = np.min([ bbox[0]+bbox[2] for bbox in bboxes])
    y2 = np.min([ bbox[1]+bbox[3] for bbox in bboxes])
    
    intersection_bbox = [x1, y1, x2-x1, y2-y1 ]
    return intersection_bbox

def read_preproc_data(hdf_filepath=None, hdf_filename=None, video_gt=None, 
                      data_path=None, ignore_no_file=False):
    if data_path is None:
        if video_gt is None:
            raise ValueError("Both data_path and video_gt are None.")
        prefix, clip = os.path.split(video_gt.path)
        prefix, video = os.path.split(prefix)
        data_path = video+'/'+clip
    
    if hdf_filepath is None:
        if hdf_filename is None:
            raise ValueError("Both hdf_filepath and hdf_filename are None.")
        data_dir, head = os.path.split(prefix)
        
        if os.path.exists(os.path.join(data_dir, 'localstorage', hdf_filename)):
            hdf_filepath = os.path.join(data_dir, 'localstorage', hdf_filename)
        else:
            hdf_filepath = os.path.join(data_dir, hdf_filename)
    
    if not os.path.exists(hdf_filepath) and ignore_no_file:
        return None
    
    with h5py.File(hdf_filepath, "r") as f:
        data = f[data_path][()]
    
    return data

def argsort_players(video_gt):
    """
    Sort players based on their (x,y) location into different pre-defined
    volleyball positions (front vs back, and middle vs exterior).
    Used for separating players into their playing position and team.
    The order will follow a clockwise view of the team, starting with left team
    and the player at front+exterior:
          6 1     7 12
        5     2 8     11
          4 3     9 10
    

    Parameters
    ----------
    video_gt : pandas Series
        DESCRIPTION.

    Returns
    -------
    sorted_idx : list
        List with sorted players indexes.

    """
    players_info = get_persons_info(video_gt)
    bboxes = players_info[['x','y','width','height']].values
    
    ## %% WIP
    # from sklearn.cluster import KMeans
    
    # 1) Splitting teams
    players_x_pos = bboxes[:,0]
    players_y_pos = bboxes[:,1]
    
    if len(bboxes) == 12:
        horizontal_sort = np.argsort(players_x_pos)
        left_team = horizontal_sort[:6]
        right_team = horizontal_sort[6:][::-1]
    else:
        # pivot = players_x_pos.mean()
        pivot = np.mean([players_x_pos.max(),players_x_pos.min()])
        left_team = np.argwhere(players_x_pos <= pivot).flatten().tolist()
        right_team = np.argwhere(players_x_pos > pivot).flatten().tolist()
        
        left_team = sorted(left_team, key=lambda idx: players_x_pos[idx])
        right_team = sorted(right_team, key=lambda idx: players_x_pos[idx], 
                            reverse=True)
        
        while len(left_team) > 6:
            right_team.append(left_team.pop())
        while len(right_team) > 6:
            left_team.append(right_team.pop())
        # labels = KMeans(n_clusters=2).fit_predict(players_x_pos[:,np.newaxis])
    
    # 2) Split front and back and use Vertical position to sort players
    sorted_idx = []
    for team in [left_team, right_team]:
        front = sorted(team[-3:], key=lambda idx: players_y_pos[idx])
        back = sorted(team[:-3], key=lambda idx: players_y_pos[idx], reverse=True)
        
        if len(front) == 0:
            front = [None, None, None]
        elif len(front) == 1:
            front = [None, front[0], None]
        elif len(front) == 2:
            front = [front[0], None, front[1]]
        
        if len(back) == 0:
            back = [None, None, None]
        elif len(back) == 1:
            back = [None, back[0], None]
        elif len(back) == 2:
            back = [back[0], None, back[1]]
        
        sorted_idx += front + back
    assert len(sorted_idx) == 12, "Missing player position mapping"
    
    return sorted_idx
