import os, math
import numpy as np

from tensorflow.keras.utils import to_categorical

from misc import utils, video_utils
from misc.data_io import get_balls_coords

def read_video_ball_flows(video_gt, hdf_filepath, timesteps=10, pad=20, 
                          res_ratio=2/3, img_size=(224,224)):
    """
    Read video flows (u and v) and crop region around ball coordinate.

    Parameters
    ----------
    video_gt : pandas Series
        Groundtruth information from video.
    hdf_filepath : string
        Path to file containing the ball cordinates annotation.
    timesteps : int, optional
        How many frame timesteps to get. The default is 10. Use None to get all.
        Analagous as stack length since flows will be stacked.
    pad : int, optional
        Pad value to add to bbox. The default is 20.
    res_ratio : float, optional
        Resize resolution ratio from HD videos to SD. The default is 2/3.
        HD: 1920x1080, SD: 1280x720

    Returns
    -------
    cropped_flows : array (timesteps, 2) + (bbox_w, bbox_h)
        cropped regions per indivs in all frames.
    """
    
    flows = read_video_flows(video_gt, timesteps=timesteps, direction='both',
                             stack=True)
    
    balls_coords = get_balls_coords(video_gt, hdf_filepath, 
                    use_center=True, interpolate=False)
    # Ball coords from Volleyball can only be center
    # Transform center into bbox
    # balls_bboxes = [utils.get_point_bbox(c[0], size=2*pad) 
    balls_bboxes = [utils.get_point_bbox(c[0], size=2) 
                    for c in balls_coords]
    
    num_flows = len(flows)
    mid_flow_num = math.floor(len(balls_bboxes)/2)
    start = mid_flow_num - math.ceil((num_flows-1)/2)
    end = 1 + mid_flow_num + math.floor((num_flows-1)/2)
    
    balls_bboxes = list(balls_bboxes)[start:end]
    
    cropped_flows = []
    for flow, ball_bbox in zip(flows, balls_bboxes):
        flow_u, flow_v = flow
        
        if ball_bbox is not None:
            if video_gt.resolution == 'hd':
                ball_bbox = [int(v*res_ratio) for v in ball_bbox]
                
            cropped_ball_u = video_utils.crop_img(flow_u, ball_bbox, pad=pad,
                                                  new_size=img_size)
            cropped_ball_v = video_utils.crop_img(flow_v, ball_bbox, pad=pad,
                                                  new_size=img_size)
        else:
            cropped_ball_u = np.zeros(img_size)
            cropped_ball_v = np.zeros(img_size)
        
        cropped_flows.append([cropped_ball_u, cropped_ball_v])
    
    return cropped_flows

def read_video_ball_frames(video_gt, hdf_filepath, timesteps=10, pad=20, 
                           img_size=(224,224)):
    """
    Read video frames and crop region around ball coordinate.

    Parameters
    ----------
    video_gt : pandas Series
        Groundtruth information from video.
    hdf_filepath : string
        Path to file containing the ball cordinates annotation.
    timesteps : int, optional
        How many frame timesteps to get. The default is 10.
    pad : int, optional
        Pad value to add to bbox. The default is 20.

    Returns
    -------
    cropped_frames : array (num_frames, bbox_w, bbox_h, 3)
        cropped regions per indivs in all frames..

    """
    frames = read_video_frames(video_gt, timesteps=timesteps)
    
    balls_coords = get_balls_coords(video_gt, hdf_filepath, 
                    use_center=True, interpolate=False)
    # Ball coords from Volleyball can only be center
    # Transform center into bbox
    # balls_bboxes = [utils.get_point_bbox(c[0], size=2*pad) 
    balls_bboxes = [utils.get_point_bbox(c[0], size=2) 
                    for c in balls_coords]
    
    num_frames = len(frames)
    mid_frame_num = math.floor(len(balls_bboxes)/2)
    start = mid_frame_num - math.ceil((num_frames-1)/2)
    end = 1 + mid_frame_num + math.floor((num_frames-1)/2)
    
    balls_bboxes = list(balls_bboxes)[start:end]
    
    if video_gt.resolution == 'hd':
        res_ratio = 2/3
        pad = int(pad/res_ratio)
    
    cropped_frames = []
    for frame, ball_bbox in zip(frames, balls_bboxes):
        if ball_bbox is not None:
            cropped_ball = video_utils.crop_img(frame, ball_bbox, pad=pad, 
                                                new_size=img_size)
        else:
            cropped_ball = np.zeros(img_size + (3,))
        
        cropped_frames.append(cropped_ball)
    
    return cropped_frames

def read_video_indivs_flows(video_gt, timesteps=10, pad=20, res_ratio=2/3, 
                            img_size=(224,224)):
    """
    Read video flows and crop the individuals using the event bounding boxes.

    Parameters
    ----------
    video_gt : pandas Series
        Groundtruth information from video.
    timesteps : int, optional
        How many frame timesteps to get. The default is 10. Use None to get all.
        Analagous as stack length since flows will be stacked.
    pad : int, optional
        Pad value to add to bbox. The default is 20.
    res_ratio : float, optional
        Resize resolution ratio from HD videos to SD. The default is 2/3.
        HD: 1920x1080, SD: 1280x720

    Returns
    -------
    indivs_flows : array (timesteps, num_indivs, 2) + (bbox_w, bbox_h)
        cropped regions per indivs in all frames.

    """
    flows = read_video_flows(video_gt, timesteps=timesteps, direction='both',
                             stack=True)
    
    event_bboxes = video_gt['event_bboxes']
    
    num_flows = len(flows)
    mid_flow_num = math.floor(len(event_bboxes)/2)
    start = mid_flow_num - math.ceil((num_flows-1)/2)
    end = 1 + mid_flow_num + math.floor((num_flows-1)/2)
    
    flows_event_bboxes = list(event_bboxes.values())[start:end]
    
    cropped_flows = []
    for flow, flow_bboxes in zip(flows, flows_event_bboxes):
        flow_u, flow_v = flow
        cropped_indivs = []
        for p_bbox in flow_bboxes:
            if video_gt.resolution == 'hd':
                p_bbox = [int(v*res_ratio) for v in p_bbox]
            cropped_indiv_u = video_utils.crop_img(flow_u, p_bbox, pad=pad,
                                                   new_size=img_size)
            cropped_indiv_v = video_utils.crop_img(flow_v, p_bbox, pad=pad,
                                                   new_size=img_size)
            cropped_indivs.append([cropped_indiv_u,cropped_indiv_v])
        
        cropped_flows.append(cropped_indivs)
    
    indivs_flows = cropped_flows
    return indivs_flows

def read_video_indivs_frames(video_gt, timesteps=10, pad=20, img_size=(224,224)):
    """
    Read video frames and crop the individuals using the event bounding boxes.

    Parameters
    ----------
    video_gt : pandas Series
        Groundtruth information from video.
    timesteps : int, optional
        How many frame timesteps to get. The default is 10.
    pad : int, optional
        Pad value to add to bbox. The default is 20.

    Returns
    -------
    indivs_frames : array (num_frames, num_indivs) + (224, 224, 3)
        cropped regions per indivs in all frames.
        Image size fixed to (224,224) to use it with vgg16.

    """
    frames = read_video_frames(video_gt, timesteps=timesteps)
    
    event_bboxes = video_gt['event_bboxes']
    
    num_frames = len(frames)
    mid_frame_num = math.floor(len(event_bboxes)/2)
    start = mid_frame_num - math.ceil((num_frames-1)/2)
    end = 1 + mid_frame_num + math.floor((num_frames-1)/2)
    
    frames_event_bboxes = list(event_bboxes.values())[start:end]
    
    if video_gt.resolution == 'hd':
        res_ratio = 2/3
        pad = int(pad/res_ratio)
    
    cropped_frames = []
    for frame, frame_bboxes in zip(frames, frames_event_bboxes):
        cropped_indivs = []
        for p_bbox in frame_bboxes:
            cropped_indiv = video_utils.crop_img(frame, p_bbox, pad=pad, 
                                                 new_size=img_size)
            cropped_indivs.append(cropped_indiv)
        
        cropped_frames.append(cropped_indivs)
    
    indivs_frames = cropped_frames
    return indivs_frames

def read_video_frames(video_gt, timesteps=10):
    
    frames_path = video_gt.video_path
    if os.path.exists(frames_path.replace('videos', 'localstorage/videos')):
        frames_path = frames_path.replace('videos', 'localstorage/videos')
    elif os.path.exists(frames_path.replace('images', 'localstorage/images')):
        frames_path = frames_path.replace('images', 'localstorage/images')
    
    ## Using frame_filepath allows reading only the required frames
    if not frames_path.endswith('.jpg'):
        frame_filepath = os.path.join(frames_path, 
                                  str(video_gt.frame)+'.jpg')
    else:
        frame_filepath = frames_path
    
    # frames = utils.read_video(video_gt.video_path)
    frames = utils.read_video(frame_filepath, temp_window=timesteps)
    
    return frames

def read_video_flows(video_gt, timesteps=10, direction='u', stack=False):
    """
    Read extracted flows from videos.

    Parameters
    ----------
    video_gt : pandas Series
        Groundtruth information from video.
    timesteps : int, optional
        How many frame timesteps to get. The default is 10. Use None to get all.
    direction : string, optional
        Read flows from which direction. The default is 'u'.
            'u' or 'x' or 'h' for horizontal motion over the x-axis
            'v' or 'y' for vertical motion over the y-axis
            'both' for reading flows from both directions
    stack : bool, optional
        Whether to stack or not the flows from both directions. 
        The default is False.

    Returns
    -------
    flows : list of optical flows
        Either 'u', 'v' or 'both' sequential flows.

    """
    flows_path = video_gt.video_path.replace('videos','flows')
    
    if os.path.exists(flows_path.replace('flows', 'localstorage/flows',)):
        flows_path = flows_path.replace('flows', 'localstorage/flows',)
    
    if direction == 'both':
        flows_u = utils.read_flows(flows_path, direction='u', 
                           temp_window=timesteps, mid_frame_num=video_gt.frame)
        flows_v = utils.read_flows(flows_path, direction='v', 
                           temp_window=timesteps, mid_frame_num=video_gt.frame)
        if not stack:
            flows = [flows_u, flows_v]
        else:
            flows = [[flow_u,flow_v] for flow_u,flow_v in zip(flows_u,flows_v)]
    else:
        flows = utils.read_flows(flows_path, direction=direction, 
                           temp_window=timesteps, mid_frame_num=video_gt.frame)
    
    return flows

def get_data(gt_split, timesteps=10, data_type='frames', stack_flows=True,
             num_classes_indiv=None, num_classes_grp=None, group_indivs=False, 
             num_indivs=None, add_ball=False, balls_hdf_filepath=None, 
             sort_players=False, crop_pad_val=20, img_size=(224,224)):
    
    if num_classes_indiv is None and num_classes_grp is None:
        raise ValueError("num_classes_indiv and num_classes_grp are both None")
    
    all_video_indivs = []
    all_video_balls = []
    #### Reading data for each video in gt_split
    for video_id, video_gt in gt_split.iterrows():
        ts = timesteps
        
        if data_type=='frames':
            video_indivs_data = read_video_indivs_frames(video_gt, 
                             timesteps=ts, pad=crop_pad_val, img_size=img_size)
        elif data_type=='flows':
            video_indivs_data = read_video_indivs_flows(video_gt,
                             timesteps=ts, pad=crop_pad_val, img_size=img_size)
            video_indivs_data = np.moveaxis(video_indivs_data, 2, -1)
            
        # all_video_indivs.append(video_indivs_data)
        all_video_indivs.append(np.asarray(video_indivs_data))
        
        if add_ball:
            if data_type=='frames':
                ball_data = read_video_ball_frames(video_gt, balls_hdf_filepath, 
                           timesteps=ts, pad=crop_pad_val, img_size=img_size)
            elif data_type=='flows':
                ball_data = read_video_ball_flows(video_gt, balls_hdf_filepath, 
                            timesteps=ts, pad=crop_pad_val, img_size=img_size)
                ball_data = np.moveaxis(ball_data, 1, -1)
            all_video_balls.append( np.asarray(ball_data) )
    
    persons_info = utils.get_persons_info(gt_split.iloc[0], prune_missing=False)
    if num_indivs is None:
        num_persons = len(persons_info)
    else:
        num_persons = num_indivs
    
    #### Processing data into network input
    X = []
    Y_indiv, Y_grp = [], []
    for i, video_data in enumerate(all_video_indivs):
        video_gt = gt_split.iloc[i]
        persons_info = utils.get_persons_info(video_gt, prune_missing=False)
        indiv_actions = persons_info.action_id.values[:num_persons]
        Y_grp.append(video_gt.grp_activity_id)
        
        # num_frames = len(video_data)
        
        sel_video_data = video_data[:,:num_persons]
        
        if sel_video_data.shape[1] < num_persons: 
            # Need to append person so all videos have the same number
            pad_val = num_persons - sel_video_data.shape[1]
            # if data_type == 'frames':
            #     pad_width = ((0,0), (0,pad_val), (0,0), (0,0), (0,0))
            # elif data_type == 'flows':
            #     print("Warning: Not tested")
            #     pad_width = ((0,0), (0,pad_val), (0,0), (0,0))
            pad_width = ((0,0), (0,pad_val), (0,0), (0,0), (0,0))
            sel_video_data = np.pad(sel_video_data, pad_width=pad_width)
        
        if sort_players:
            sorted_idx = utils.argsort_players(video_gt)
            sorted_idx = [ (idx if idx is not None else 11) 
                          for idx in sorted_idx ]
            sel_video_data = sel_video_data[:,sorted_idx]
            indiv_actions = indiv_actions[sorted_idx]
        
        if add_ball:
            ball_data = all_video_balls[i]
            sel_video_data = np.concatenate([sel_video_data,
                              ball_data[:,np.newaxis,:,:]], axis=1)
        
        # if num_frames < timesteps: # Can assume always correct ts
        # if timesteps < num_frames:
        
        # Shape so far: (timesteps, num_objs, **img_dims)
        sel_video_data = np.swapaxes(sel_video_data, 0, 1)
        
        X.append(sel_video_data)
        Y_indiv += indiv_actions.tolist()
    
    X = np.asarray(X)
    if not group_indivs:
        X = X.reshape((-1,) + X.shape[2:])
    
    #### Stacking flows
    if data_type == 'flows' and stack_flows:
        ## Stacking flows alternately u+v: (u_1 + v_1 + u_2 + v_2 + ...)) 
        X = np.moveaxis(X, -4, -2)
        ## Other option would be to do it all u + all v
        # X = np.moveaxis(X, -4, -1)
        X = X.reshape(X.shape[:-2] + (-1,))
    
    #### Preparing Y output
    ### Replacing dummy persons action by most common
    Y_indiv = np.asarray(Y_indiv)
    most_common = np.bincount(Y_indiv[Y_indiv!=-1]).argmax()
    Y_indiv[Y_indiv==-1] = most_common
    
    Y = []
    if num_classes_grp is not None:
        Y_grp = to_categorical(Y_grp, num_classes_grp)
        Y.append(Y_grp)
    if num_classes_indiv is not None:
        indiv_ova_class = None # One Vs All class (Not needed here)
        if indiv_ova_class is None:
            Y_indiv = to_categorical(Y_indiv, num_classes_indiv)
        else:
            num_classes_indiv = 2
            bin_Y_indiv = np.zeros_like(Y_indiv)
            bin_Y_indiv[Y_indiv==indiv_ova_class] = 1
            Y_indiv = to_categorical(bin_Y_indiv, num_classes_indiv)
            
        
        if num_classes_grp is None:
            Y.append(Y_indiv)
        else:
            Y += [ indiv for indiv in 
                  # Y_indiv.reshape((num_persons, -1, num_classes_indiv)) ]
                  Y_indiv.reshape((-1, num_persons, 
                                   num_classes_indiv)).transpose((1,0,2)) ]
    
    if len(Y) == 1:
        Y = Y[0]
    
    return X, Y