import numpy as np
import json, glob, os, h5py
import warnings

from tensorflow.keras.utils import to_categorical

from .utils import (get_persons_info, read_video, get_video_center_bbox,
    transform_bbox, get_bboxes, estimate_motion, apply_cam_motion,
    video_poses_to_coords, video_coords_to_poses, merge_bboxes, 
    read_preproc_data, argsort_players)
# from . import detector

TORSO, LEFT_HAND, RIGHT_HAND, LEFT_LEG, RIGHT_LEG = 0,1,2,3,4

# OpenPose body parts

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25
POSE_BODY_25_BODY_PARTS = ["Nose","Neck","RShoulder","RElbow","RWrist",
   "LShoulder","LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip",
   "LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel",
   "RBigToe","RSmallToe","RHeel"]

POSE_BODY_25_BODY_PARTS_COARSE = [TORSO, TORSO, RIGHT_HAND, RIGHT_HAND, 
  RIGHT_HAND, LEFT_HAND, LEFT_HAND, LEFT_HAND, TORSO, RIGHT_LEG, RIGHT_LEG,
  RIGHT_LEG,LEFT_LEG,LEFT_LEG,LEFT_LEG,TORSO,TORSO,TORSO,TORSO,LEFT_LEG,
  LEFT_LEG,LEFT_LEG,RIGHT_LEG,RIGHT_LEG,RIGHT_LEG]
POSE_BODY_25_BODY_PARTS_COARSE_TEXT = ["TORSO", "TORSO", "RIGHT_HAND", 
   "RIGHT_HAND", "RIGHT_HAND", "LEFT_HAND", "LEFT_HAND", "LEFT_HAND", 
   "TORSO", "RIGHT_LEG", "RIGHT_LEG","RIGHT_LEG","LEFT_LEG","LEFT_LEG",
   "LEFT_LEG","TORSO","TORSO","TORSO","TORSO","LEFT_LEG","LEFT_LEG","LEFT_LEG",
   "RIGHT_LEG","RIGHT_LEG","RIGHT_LEG"]


ALPHA_COCO_17_BODY_PARTS = ["Nose","LEye","REye","LEar","REar","LShoulder",
    "RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee",
    "RKnee","LAnkle","RAnkle"]
ALPHA_COCO_17_BODY_PARTS_COARSE = [TORSO,TORSO,TORSO,TORSO,TORSO,LEFT_HAND,
     RIGHT_HAND,LEFT_HAND,RIGHT_HAND,LEFT_HAND,RIGHT_HAND,LEFT_LEG,RIGHT_LEG,
     LEFT_LEG,RIGHT_LEG,LEFT_LEG,RIGHT_LEG]

SUBSET_3_JOINTS  = ['Nose','Neck','MidHip']
SUBSET_6_JOINTS   = ['Neck','MidHip', 'RWrist','LWrist','RAnkle','LAnkle']
SUBSET_6b_JOINTS  = ['Neck','MidHip', 'RElbow','LElbow','RKnee','LKnee']
SUBSET_7_JOINTS  = SUBSET_3_JOINTS + ['RWrist','LWrist','RAnkle','LAnkle']
SUBSET_7b_JOINTS  = ['Nose','Neck','MidHip', 'RElbow','LElbow','RKnee','LKnee']
SUBSET_8_JOINTS  = ['Nose','LShoulder','RShoulder','LHip','RHip',
                    'RElbow','LElbow','RKnee','LKnee']
SUBSET_9_JOINTS  = SUBSET_7_JOINTS + ['RElbow','LElbow']
SUBSET_11_JOINTS = SUBSET_9_JOINTS + ['RKnee','LKnee']
# Same as SBU_15_BODY_PARTS \/
SUBSET_13_JOINTS = ["Nose","LShoulder", "RShoulder","LElbow","RElbow",
            "LWrist","RWrist","LHip","RHip","LKnee", "Rknee","LAnkle","RAnkle"]
SUBSET_15_JOINTS = ["Nose","Neck","MidHip","LShoulder","LElbow","LWrist","RShoulder",
             "RElbow","RWrist","LHip","LKnee","LAnkle","RHip","RKnee","RAnkle"]

SUBSET_17_JOINTS = ALPHA_COCO_17_BODY_PARTS.copy()

## SUBSET_UPPER_BODY
UPPER_2_JOINTS = ["RWrist", "LWrist"]
UPPER_2b_JOINTS = ["RElbow", "LElbow"]
UPPER_3_JOINTS = ["Neck", "RWrist", "LWrist"]
UPPER_4_JOINTS = ["RElbow", "RWrist", 
                  "LElbow", "LWrist"]
UPPER_5_JOINTS = ["Neck",  "RElbow", "RWrist", 
                  "LElbow", "LWrist"]
UPPER_6_JOINTS = ["RShoulder", "RElbow", "RWrist", 
                  "LShoulder", "LElbow", "LWrist"]
UPPER_6b_JOINTS = ["Neck", "MidHip", "RElbow", "RWrist", 
                                     "LElbow", "LWrist"]
UPPER_8_JOINTS = ["Neck"] + UPPER_6_JOINTS + ["MidHip"]


def filter_joints(person, selected_joints, joint_indexing=POSE_BODY_25_BODY_PARTS):
    # joints_mask = np.isin(joint_indexing, selected_joints)
    ## Mask w/ indexes is used to match the order of the joints with selection
    joints_mask = [ joint_indexing.index(joint) for joint in selected_joints]
    # selected_parts = np.array(joint_indexing)[joints_mask]
    # joints_mask = joints_mask[:-1] # Skipping "Background"
    
    ## At previous version the it was updating the object directly
    # selected_coords = np.asarray(person['coords'])[joints_mask]
    # person['coords'] = selected_coords
    
    selected_coords = person[joints_mask]
    
    
    # selected_confs = np.array(person['confs'])[joints_mask]
    # person['confs'] = selected_confs
    
    # return person
    return selected_coords

def parse_json(json_filepath, prune=True, conf_thres=0.05):
    with open(json_filepath) as json_file:
        frame_data = json.load(json_file)
    
    people = []
    for person in frame_data['people']:
        pose_keypoints_2d = person['pose_keypoints_2d']
        confidences = pose_keypoints_2d[2::3]
        if prune and np.mean(confidences) < conf_thres: # 0.15
            # confidences = np.asarray(pose_keypoints_2d[2::3])
            # print(confidences[confidences.nonzero()].mean())
            continue
        coords_x = pose_keypoints_2d[0::3]
        coords_y = pose_keypoints_2d[1::3]
        coords = np.array([coords_x, coords_y]).T
        ### TODO return coords np.array/list
        per = {}
        per['coords'] = coords
        per['confs'] = confidences
        people.append(per)
    
    return people

def parse_json_alpha(json_filepath, prune=True, conf_thres=0.05):
    with open(json_filepath) as json_file:
        frame_data = json.load(json_file)
    
    people = []
    # for person in frame_data['people']:
    for person in frame_data:
        
        bbox = person['box']
        area = bbox[2]*bbox[3]
        if area > 10**5: # Too big
            continue
        
        # person_score = person['score']
        # # if person_score < 1.:
        # #     continue
        
        pose_keypoints_2d = person['keypoints']
        confidences = pose_keypoints_2d[2::3]
        if prune and np.mean(confidences) < conf_thres:
            continue
        coords_x = pose_keypoints_2d[0::3]
        coords_y = pose_keypoints_2d[1::3]
        coords = np.array([coords_x, coords_y]).T
        ### TODO return coords np.array/list
        per = {}
        per['coords'] = coords
        per['confs'] = confidences
        people.append(per)
    
    return people

def apply_NTU_normalization(video_poses, pose_style='OpenPose'):
    """ From "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"
        > We translate them to the body coordinate system with its origin on 
        the “middle of the spine” joint (number 2 in Figure 1), followed by a 
        3D rotation to fix the X axis parallel to the 3D vector from 
        “right shoulder” to “left shoulder”, and Y axis towards the 3D vector 
        from “spine base” to “spine”. The Z axis is fixed as the new X × Y. In 
        the last step of normalization, we scale all the 3D points based on the
        distance between “spine base” and “spine” joints.
        > In the cases of having more than one body in the scene, we transform 
        all of them with regard to the main actor’s skeleton.
    
        Obs: Since OpenPose and SBU do not have the joints “middle of the spine”
        and “spine”, 'MidHip' and 'Neck' are respectively used instead.
    """
    print("WARNING: Not tested yet! Needs visualization")
    
    if pose_style == 'OpenPose':
        joint_indexing = POSE_BODY_25_BODY_PARTS
    else:
        raise NotImplementedError("Invalid pose_style: "+pose_style)
    
    # No 'Middle_of_the_spine', so will avg 'MidHip' and 'Neck'
    # middle_joint_idx = joint_indexing.index('Middle_of_the_spine')
    spine_base_joint_idx = joint_indexing.index('MidHip')
    spine_joint_idx = joint_indexing.index('Neck')
    left_shoulder_joint_idx = joint_indexing.index('LShoulder')
    right_shoulder_joint_idx = joint_indexing.index('RShoulder')
    
    num_dim = len(video_poses[0][0]['coords'][0])
    
    normalized_video_poses = []
    for frame_idx, frame_poses in enumerate(video_poses):
        p1_coords = frame_poses[0]['coords']
        p1_spine_base_joint = p1_coords[spine_base_joint_idx]
        p1_spine_joint = p1_coords[spine_joint_idx]
        p1_left_shoulder_joint = p1_coords[left_shoulder_joint_idx]
        p1_right_shoulder_joint = p1_coords[right_shoulder_joint_idx]
        p1_middle_joint = np.mean([p1_spine_base_joint,p1_spine_joint], axis=0)
        
        new_origin = p1_middle_joint
        # scale_val = .5 / np.linalg.norm(p1_spine_base_joint - p1_spine_joint)
        
        y = p1_spine_joint - p1_spine_base_joint
        y = y/np.linalg.norm(y)
        
        x = p1_left_shoulder_joint - p1_right_shoulder_joint
        x = x/np.linalg.norm(x)
        
        if num_dim == 2:
            rotation_matrix = np.array([x,y])
        if num_dim == 3:
            z = np.cross(x, y)
            z = z/np.linalg.norm(z)
            
            x = np.cross(y,z)
            x = x/np.linalg.norm(x)
            
            rotation_matrix = np.array([x,y,z])
        
        normalized_frame_poses = []
        for person in frame_poses:
            if np.count_nonzero(person) > 0: # Checking if it is not a dummy
                translated = person['coords'] - new_origin
                rotated = np.dot(rotation_matrix, translated.T).T
                scaled = rotated# * scale_val
            else:
                scaled = person['coords'].copy()
            normalized_frame_poses.append(scaled)
        
        if np.isnan(normalized_frame_poses).any():
            if normalized_video_poses == []:
                normalized_frame_poses = np.zeros_like(normalized_frame_poses)
            else:
                normalized_frame_poses = normalized_video_poses[-1]
        
        # normalized_person = {'coords': scaled, 'confs': person['confs'].copy()}
        normalized_frame_poses = [ {'coords': coords } 
                                  for coords in normalized_frame_poses]
        normalized_video_poses.append(normalized_frame_poses)
        
    return normalized_video_poses

def apply_center_normalization(video_poses, new_origin='skels_center',
                               return_estimations=False):
    """
    Normalize poses based on estimated camera motion and estimated center.
    First, translates all coords per frame to remove/reduce change because of
    camera motion.
    Then translate all video coords based on a new point of origin estimated 
    from ast the mean from all coordinates.

    Parameters
    ----------
    video_poses : list
        List of frame poses with skeletons for all individuals.
    new_origin : str, optional
        Where to set the new origin. Set None for no translation.
        The default is 'skels_center'.
        

    Returns
    -------
    norm_video_poses : list
        video_poses after coordinates normalization.

    """
    # skeletons = np.asarray([ [ person['coords'] for person in frame_poses ]
    #                         for frame_poses in video_poses])
    skeletons = np.array(video_poses)
    
    # if not np.any(np.all(skeletons, axis=-1)):
    if not np.any(skeletons): # All zeros
        ret = skeletons
        if return_estimations:
            cumsum_cam_motions = np.zeros((len(skeletons)-1,2))
            video_center = np.zeros(2)
            ret = (skeletons, cumsum_cam_motions, video_center)
        
        return ret
    
    cumsum_cam_motions = []
    prvs_frame_skels = skeletons[0]
    for frame_skels in skeletons[1:]:
        # print('+'*40)
        # print(frame_skels[:,0])
        
        motions = []
        for skels, prvs_skels in zip(frame_skels, prvs_frame_skels):
            motion = skels - prvs_skels
            # print(motion)
            non_zero_idx = np.logical_and(np.all(skels, axis=1), 
                                          np.all(prvs_skels, axis=1))
            motions += motion[non_zero_idx].tolist()
        
        # print(np.any(np.all(motions, axis=-1)))
        # print(np.all(motions, axis=-1))
        # print(motions[0])
        if motions != [] and np.any(np.all(motions, axis=-1)):
            motions = np.asarray(motions)
            # print(motions.shape)
            # print(motions[0])
            
            x_motions = motions[..., 0]
            y_motions = motions[..., 1]
            
            # print(x_motions.mean(), x_motions.std())
            # print(y_motions.mean(), y_motions.std())
            
            if x_motions.std() > 0:
                x_motions = x_motions[ abs(x_motions-x_motions.mean())
                                      < 2*x_motions.std()]
            if y_motions.std() > 0:
                y_motions = y_motions[ abs(y_motions-y_motions.mean())
                                      < 2*y_motions.std()]
            
            cam_motion = [x_motions.mean(axis=0), y_motions.mean(axis=0)]
        else:
            cam_motion = [0,0]
        # print("cam_motion", cam_motion)
        cumsum_cam_motions.append(cam_motion)
        
        frame_skels[np.all(frame_skels, axis=-1)] -= cam_motion
        prvs_frame_skels = frame_skels
    
    # Alternative: use center based on groundtruth bounding boxes
    if new_origin == 'skels_center':
        video_center = skeletons[np.all(skeletons, axis=-1)].mean(axis=0)
        # print(skeletons[np.all(skeletons, axis=-1)])
        # print(skeletons[np.all(skeletons, axis=-1)].shape)
        skeletons[np.all(skeletons, axis=-1)] -= video_center
    else:
        video_center = [0,0]
    
    # norm_video_poses = [
    #     [ {'coords': person_pose} for person_pose in frame_poses] 
    #     for frame_poses in skeletons ]
    norm_video_poses = skeletons
    
    ret = norm_video_poses
    if return_estimations:
        ret = (norm_video_poses, cumsum_cam_motions, video_center)
    
    return ret

def apply_middle_cam_motion(video_poses, cam_motions):
    num_frames = len(video_poses)
    
    if cam_motions.shape[0] != num_frames - 1:
        # Need to use only the middle cam_motions
        mid_motions = cam_motions.shape[0]//2
        central_slice = slice(mid_motions - num_frames//2,
                              mid_motions + num_frames//2 + num_frames%2 - 1)
        cam_motions = cam_motions[central_slice]
        
    
    mid_frm = num_frames//2
    
    # first_half = apply_cam_motion(video_poses[:mid_frm][::-1], 
    #                                     -cam_motions[:mid_frm-1][::-1], 
    #                                     'coords', poses=True)[::-1]
    # second_half = apply_cam_motion(video_poses[mid_frm-1:], 
    #                                     cam_motions[mid_frm-1:], 
    #                                     'coords', poses=True)[1:]
    first_half = apply_cam_motion(video_poses[:mid_frm+1][::-1], 
                                        -cam_motions[:mid_frm][::-1], 
                                        'coords', poses=False)[::-1]
    second_half = apply_cam_motion(video_poses[mid_frm:], 
                                        cam_motions[mid_frm:], 
                                        'coords', poses=False)[1:]
    norm_video_poses = np.asarray(first_half + second_half)
    return norm_video_poses

def apply_norm(video_poses, norm, video_gt=None):
    """
        Possible normalizations:
            center: remove cam motion and translate all coords according to
                computed center.
            cam_motion: just remove cam motion (keep first frame same, and shift 
                subsequent)
            middle_cam_motion: remove cam motion, but around midle frame
            event_center: remove cam motion and translate all coords according to
                center of bounding boxes in the middle frame (event frame)
            *skels_center: similar to center, but use pre-computed values using
                center of skeletons only (not using the ball coordinates)
            NTU: Not used, untested.
    """
    # if video_poses == [] or np.asarray(video_poses).size == 0:
    # if video_poses == [] or not any(video_poses):
    if ((isinstance(video_poses, list) and not any(video_poses)) or 
            (isinstance(video_poses, np.ndarray) and video_poses.size == 0)):
        return video_poses
    norm_video_poses = None
    
    
    if norm == 'center':
        norm_video_poses = apply_center_normalization(video_poses)
    elif norm == 'cam_motion':
        norm_video_poses = apply_center_normalization(video_poses, 
                                                      new_origin=None)
    elif norm == 'middle_cam_motion':
        # video_frames = read_video(video_gt.video_path)
        # cam_motions = estimate_motion(video_frames) # clean_outliers=True?
        cam_motions = read_preproc_data(hdf_filename='cam_motions.hdf5', 
                                              video_gt=video_gt)
        
        norm_video_poses = apply_middle_cam_motion(video_poses, cam_motions)
    elif norm == 'event_center':
        # video_frames = read_video(video_gt.video_path)
        # cam_motions = estimate_motion(video_frames) # clean_outliers=True?
        cam_motions = read_preproc_data(hdf_filename='cam_motions.hdf5', 
                                              video_gt=video_gt)
        
        norm_video_poses = apply_middle_cam_motion(video_poses, cam_motions)
        
        persons_info = get_persons_info(video_gt)
        persons_bboxes = persons_info[['x','y','width','height']].values
        event_roi = merge_bboxes(persons_bboxes)
        event_center = [event_roi[0]+event_roi[2]//2, event_roi[1]+event_roi[3]//2]
        
        # video_coords = np.asarray(video_poses_to_coords(norm_video_poses))
        # video_coords[np.all(video_coords, axis=-1)] -= event_center
        # norm_video_poses = video_coords_to_poses(video_coords)
        norm_video_poses[np.all(norm_video_poses, axis=-1)] -= event_center
    elif norm == 'skels_center':
        # video_frames = read_video(video_gt.video_path)
        # cam_motions = estimate_motion(video_frames) # clean_outliers=True?
        cam_motions = read_preproc_data(hdf_filename='cam_motions-skels.hdf5', 
                                              video_gt=video_gt)
        
        # norm_video_poses = apply_middle_cam_motion(video_poses, cam_motions)
        norm_video_poses = apply_cam_motion(video_poses, cam_motions, 'coords', 
                                            poses=False)
        norm_video_poses = np.asarray(norm_video_poses)
        
        skels_center = read_preproc_data(hdf_filename='center-skels.hdf5', 
                                              video_gt=video_gt)
        
        # video_coords = np.asarray(video_poses_to_coords(norm_video_poses), 
        #                           dtype='float32')
        # video_coords[np.all(video_coords, axis=-1)] -= skels_center
        # norm_video_poses = video_coords_to_poses(video_coords)
        norm_video_poses[np.all(norm_video_poses, axis=-1)] -= skels_center
    elif norm == 'NTU':
        norm_video_poses = apply_NTU_normalization(video_poses)
    else:
        raise ValueError("Invalid norm: " + norm)
    
    return norm_video_poses

def interpolate_coords(coords, poses=False):
    if poses:
        coords = video_poses_to_coords(coords)
    
    coords = np.asarray(coords)
    
    num_frames, num_objs, num_dim = coords.shape
    all_frames_idx = np.arange(num_frames)
    
    # gt_frames_mask = np.any(coords, axis=(1,2))
    # gt_frames_idx = all_frames_idx[gt_frames_mask]
    # interp_coords = np.interp(all_frames_idx, gt_frames_idx, fp, 
    #                    left=0, right=0)
    
    interp_coords = []
    coords = coords.transpose((1,0,2))
    for obj_coords in coords:
        gt_frames_mask = np.any(obj_coords, axis=1)
        
        if gt_frames_mask.sum() <= 1: # Not enough to interpolate
            interp_obj_coords = obj_coords
        else:
            gt_frames_idx = all_frames_idx[gt_frames_mask]
            
            interp_obj_coords = []
            for dim_idx in range(num_dim):
                fp = [ c[dim_idx] for c in obj_coords[gt_frames_mask] ]
                interp = np.interp(all_frames_idx, gt_frames_idx, fp, 
                                    left=0, right=0)
                interp_obj_coords.append(interp)
            interp_obj_coords = np.asarray(interp_obj_coords).T
            # interp_obj_coords = np.round(interp_obj_coords, 0).T
        
        interp_coords.append(interp_obj_coords)
    interp_coords = np.transpose(interp_coords, (1,0,2))
    
    if poses:
        interp_coords = np.asarray(video_coords_to_poses(interp_coords))
    
    return interp_coords

def add_noise(coords, std=1, dropout_rate=0, resolution='sd'):
    noisy_coords = np.array(coords)
    if std > 0:
        if resolution == 'hd':
            std *= 1.5
        
        # noise = np.random.normal(0, std, coords.shape)
        sqrt_2 = 1.4142135623730951
        noise = np.random.normal(0, std/sqrt_2, coords.shape)
        # divide by square root of 2 to limit vector magnitude based on std
        
        non_zero_idx = np.all(coords, axis=-1)
        noisy_coords[non_zero_idx] += noise[non_zero_idx]
    
    if dropout_rate > 0:
        dropout_mask = np.random.binomial(1, dropout_rate, coords.shape[:-1])
        dropout_mask = dropout_mask.astype(bool)
        noisy_coords[dropout_mask] *= 0
    
    return noisy_coords

def filter_poses_roi(video_poses, video_rois, dilate=False, perc_thres=.5):
    """
    Filter video poses by checking at each frame if the pose is majoritarily
    inside the regions of interest (rois) at each frame.
    Assumes there is only one roi per frame.

    Parameters
    ----------
    video_poses : list
        list with shape (num_frames, num_poses, num_joints, dim).
    video_rois : list
        List of bounding boxes, one per frame, indicating the region of interest.
    dilate : bool, optional
        Wether to dilate the bouding box. The default is False.
    perc_thres : float, optional
        Percentage of joints inside roi to be used as threshold to determine 
        it is inside. 
        The default is .5.

    Returns
    -------
    roi_poses : list
        Filtered video poses, with those that were inside rois.

    """
    roi_poses = []
    for frame_poses, roi_bbox in zip(video_poses, video_rois):
        if roi_bbox != []:
            x1, y1 = roi_bbox[:2]
            x2 = x1 + roi_bbox[2]
            y2 = y1 + roi_bbox[3]
        else:
            # print("Warning: no roi_bbox", roi_bbox)
            x1, y1 = 0, 0
            x2, y2 = 9999, 9999
        if dilate:
            x1 -= 10
            y1 -= 10
            x2 += 10
            y2 += 10
        is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
        frame_roi_poses = []
        for pose in frame_poses:
            pose_coords = pose['coords']
            # pose_coords = pose
            
            num_valid = np.sum(np.all(pose_coords, axis=1))
            if num_valid > 0: # A joints minimum can be set here
                n_hits = np.apply_along_axis(is_inside, 1, pose_coords).sum()
                hits_perc = n_hits/num_valid
                if hits_perc > perc_thres:
                    frame_roi_poses.append(pose)
        roi_poses.append(frame_roi_poses)
    
    # if video_rois == []: # no filter if no event_rois?
        # roi_poses = video_poses
    
    return roi_poses

def prune_bodies_central_bboxes(video_poses, bboxes, metric_name='num_joints', 
                                window_size=1, metric_threshold=0.99):
    """
    Keep only those bodies that match the provided bounding boxes.
    Match is made based on the chosen metric.

    Parameters
    ----------
    video_poses : list
        Tracked poses for all frames. It assumes there is body consistency for 
        the poses under the same body index.
    bboxes : list
        Persons bouding boxes information.
    metric_name : string, optional
        Which metric to use. The default is 'num_joints'.
        'num_joints': percentage of joints inside bounding box
        'central': distance to center of mass
    window_size : int, optional
        Number of frames around center to perform matching. The default is 3.
    metric_threshold : int, optional
        Threshold for metric. If above, will consider the pose is missing.
        The default is 0.99 for 'num_joints' and 100 for 'central'.
        'num_joints': percentage of joints misses accepted
        'central': max distance accepted

    Returns
    -------
    pruned_video_poses : list
        Selected poses for all frames.

    """
    if metric_name == 'central' and metric_threshold < 1:
        metric_threshold = 100
    
    max_num_ppl = np.max([ len(frame_poses) for frame_poses in video_poses])
    bodies_coords = [ [] for _ in range(max_num_ppl) ]
    
    # for frame_poses in video_poses:
    center = len(video_poses)//2
    # central_slice = slice(20-window_size//2,21+window_size//2)
    central_slice = slice(center - window_size//2, center + 1 + window_size//2)
    for frame_poses in video_poses[central_slice]:
        for pose_idx, pose in enumerate(frame_poses):
            bodies_coords[pose_idx].append(pose['coords'].tolist())
    
    central_points = []
    for bbox in bboxes:
        bbox_central_point = [bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2]
        central_points.append(bbox_central_point)
    # central_point = np.mean(central_points, axis=0)
    
    ### Compute Metric
    metrics = []
    for body_coords in bodies_coords:
        metric = []
        # body = np.asarray(body_coords)
        if metric_name == 'num_joints':
            for bbox in bboxes:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = x1 + bbox[2]
                y2 = y1 + bbox[3]
                is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
                misses_perc = []
                for frame_coord in body_coords:
                    num_valid = np.sum(np.all(frame_coord, axis=1))
                    if num_valid == 0:
                        misses_perc.append(1.)
                        continue
                    n_hits = np.apply_along_axis(is_inside,1,frame_coord).sum()
                    misses_perc.append(1 - n_hits/num_valid)
                metric.append(np.min(misses_perc))
        elif metric_name == 'central':
            body = np.asarray(body_coords)
            for central_point in central_points:
                ### Distance to body center
                # central_idx = 8 # "MidHip"
                # dist_central = np.linalg.norm(body[:,central_idx] - central_point, axis=1).mean()
                
                ### Distance to all joints
                joints_distance = np.linalg.norm(body - central_point, axis=2)
                # dist_central = joints_distance.mean()
                # dist_central = np.median(joints_distance)
                dist_central = np.min(joints_distance)
                
                metric.append(dist_central)
        metrics.append(metric)
    metrics = np.array(metrics)
    
    sorted_args = np.argsort(metrics, axis=None)
    bbox_ids_matched = []
    skel_ids_matched = []
    for flat_arg in sorted_args:
        skel_id, bbox_id = np.unravel_index(flat_arg, metrics.shape)
        if (len(skel_ids_matched) == len(bboxes) or 
                metrics[skel_id,bbox_id] > metric_threshold):
            break
        if skel_id not in skel_ids_matched and bbox_id not in bbox_ids_matched:
            skel_ids_matched.append(skel_id)
            bbox_ids_matched.append(bbox_id)
    
    # skel_ids_matched = [17, 13,  9, 2, 5, 4, 3, 19, 0,23, 12]
    # bbox_ids_matched = [ 7,  5, 10, 8, 0, 2, 1,  4, 3, 9,  6]
    
    sorted_idx = np.argsort(bbox_ids_matched)
    skel_ids_matched = np.array(skel_ids_matched)[sorted_idx]
    bbox_ids_matched = np.array(bbox_ids_matched)[sorted_idx]
    # print([ metrics[x] for x in zip(skel_ids_matched, bbox_ids_matched) ])
    # print(skel_ids_matched)
    # print(bbox_ids_matched)
    
    null_joints_body = np.zeros_like(video_poses[0][0]['coords'])
    null_pose = {'coords': null_joints_body}
    
    pruned_video_poses = []
    for frame_poses in video_poses:
        pruned_frame_poses = []
        # This 'for' guarantees the skel order is the same as bbox order
        for bbox_id in range(len(bboxes)):
            if bbox_id in bbox_ids_matched:
                pose_idx=skel_ids_matched[np.argmax(bbox_ids_matched==bbox_id)]
                pruned_frame_poses.append(frame_poses[pose_idx])
            else:
                pruned_frame_poses.append(null_pose.copy())
                # print("Missing bbox_idx:", bbox_idx)
        # for pose_idx in skel_ids_matched:
        #     pruned_frame_poses.append(frame_poses[pose_idx])
        pruned_video_poses.append(pruned_frame_poses)
    
    return pruned_video_poses

def prune_bodies_event_rois(video_poses, video_rois, dilate=True, 
                    perc_thres=.5, top_k=10, sort=False, min_hits_perc=None):
    num_bodies = len(video_poses[0])
    bodies_hits = [0] * num_bodies

    for frame_poses, roi_bbox in zip(video_poses, video_rois):
        if roi_bbox != []:
            x1, y1 = roi_bbox[:2]
            x2 = x1 + roi_bbox[2]
            y2 = y1 + roi_bbox[3]
        else:
            # print("Warning: no roi_bbox", roi_bbox)
            x1, y1 = 0, 0
            x2, y2 = 9999, 9999
        if dilate:
            x1 -= 5
            y1 -= 10
            x2 += 5
            y2 += 10
        is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
        
        for body_idx in range(num_bodies):
            pose_coords = frame_poses[body_idx]['coords']
            
            num_valid = np.sum(np.all(pose_coords, axis=1))
            if num_valid > 0: # A joints minimum can be set here
                n_hits = np.apply_along_axis(is_inside, 1, pose_coords).sum()
                hits_perc = n_hits/num_valid
                if hits_perc > perc_thres:
                    bodies_hits[body_idx] += 1
    
    bodies_hits = np.asarray(bodies_hits)
    # print(bodies_hits)
    
    bodies_to_keep = np.sort(np.argsort(bodies_hits)[-top_k:])
    
    if min_hits_perc is not None:
        min_hits = len(video_rois)*min_hits_perc
        bodies_to_keep = [ body_idx for body_idx in bodies_to_keep 
                          if bodies_hits[body_idx] >= min_hits]
    
    if sort:
        sorted_idx = bodies_hits[bodies_to_keep].argsort()[::-1]
        bodies_to_keep = np.asarray(bodies_to_keep)[sorted_idx]
        # print(np.sort(bodies_hits)[::-1])
    
    pruned_poses = [ [ frame_poses[body_idx] for body_idx in bodies_to_keep]
                    for frame_poses in video_poses]
    
    return pruned_poses

def prune_bodies_event_bboxes(video_poses, event_bboxes, dilate=True, 
                        perc_thres=.75, top_k = None, sort=True, debug=False):
    num_frames = len(video_poses)
    num_bodies = len(video_poses[0])
    # bodies_hits = [ [] for _ in range(num_bodies) ]
    # bodies_hits = numpy.zeros((num_bodies, num_frames), dtype=bool)
    bodies_hits = np.zeros(num_bodies, dtype=int)
    bodies_area_inter = np.zeros(num_bodies, dtype=int)
    
    for frame_idx, bboxes in event_bboxes.items():
        if frame_idx >= num_frames:
            continue
        frame_poses = video_poses[frame_idx]

        for roi_bbox in bboxes:
            x1, y1 = roi_bbox[:2]
            x2 = x1 + roi_bbox[2]
            y2 = y1 + roi_bbox[3]
            if dilate:
                x1 -= 5
                y1 -= 10
                x2 += 5
                y2 += 10
            is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
            
            for body_idx in range(num_bodies):
                pose_coords = frame_poses[body_idx]['coords']
                
                num_valid = np.sum(np.all(pose_coords, axis=1))
                if num_valid > 0: # A joints minimum can be set here
                    n_hits = np.apply_along_axis(is_inside, 1, pose_coords).sum()
                    hits_perc = n_hits/num_valid
                    if hits_perc > perc_thres:
                        bodies_hits[body_idx] += 1
                        body_bbox = compute_bbox(pose_coords)
                        area_inter = compute_area_intersection(roi_bbox,
                                                               body_bbox)
                        bodies_area_inter[body_idx] += area_inter
    
    # print(bodies_hits)
    # print(sorted(bodies_hits))
    if top_k is None:
        top_k = np.max([ len(bboxes) for bboxes in event_bboxes.values() ])
        top_k = min(top_k, 12) # it does not make sense to have more than 12 players
    
    
    # bodies_to_keep = sorted(np.argsort(bodies_hits)[-top_k:])
    # hits_thres = len(event_bboxes)*.25
    # bodies_to_keep = np.asarray([ body_idx for body_idx in bodies_to_keep
    #                   if bodies_hits[body_idx] >= hits_thres])
    
    # bodies_to_keep = sorted(np.argsort(bodies_hits)[-top_k:])
    hits_thres = len(event_bboxes)*.25
    bodies_to_keep = np.asarray([ body_idx 
                             for body_idx, body_hits in enumerate(bodies_hits)
                             if body_hits >= hits_thres])
    # print(num_frames*.1, len(event_bboxes)*.25, len(event_bboxes))
    
    if debug:
        print("bodies_to_keep:", bodies_to_keep)
        # print(sorted(bodies_hits))
        # print(bodies_area_inter)
        print("bodies_hits:", [ bodies_hits[i] for i in bodies_to_keep ])
        print("bodies_area_inter:", [ bodies_area_inter[i] for i in bodies_to_keep ])
    
    if sort and bodies_to_keep.size > 0:
        # sorted_idx = bodies_hits[bodies_to_keep].argsort()[::-1]
        # bodies_to_keep = bodies_to_keep[sorted_idx]
        # print(np.sort(bodies_hits)[::-1])
        
        sorted_idx = bodies_area_inter[bodies_to_keep].argsort()[::-1]
        bodies_to_keep = bodies_to_keep[sorted_idx]
        
        bodies_to_keep = bodies_to_keep[:top_k]
    
    ### TODO apply sort to match bbox order
        # Based on prune_bodies_central_bboxes
    
    if debug:
        print("After sorting...")
        print("bodies_to_keep:", bodies_to_keep)
        # print(sorted(bodies_hits))
        # print(bodies_area_inter)
        print("bodies_hits:", [ bodies_hits[i] for i in bodies_to_keep ])
        print("bodies_area_inter:", [ bodies_area_inter[i] for i in bodies_to_keep ])
    
    pruned_poses = [ [ frame_poses[body_idx] for body_idx in bodies_to_keep]
                    for frame_poses in video_poses]
    
    return pruned_poses

def match_bodies_event_bboxes(video_poses, event_bboxes, dilate=False, 
                        perc_thres=.75, debug=False):
    num_frames = len(video_poses)
    num_bodies = len(video_poses[0])
    num_bboxes = len(list(event_bboxes.values())[0])
    # bodies_hits = [ [] for _ in range(num_bodies) ]
    # bodies_hits = numpy.zeros((num_bodies, num_frames), dtype=bool)
    bodies_hits = np.zeros((num_bodies,num_bboxes), dtype=int)
    bodies_area_inter = np.zeros((num_bodies,num_bboxes), dtype=int)
    
    for frame_idx, bboxes in event_bboxes.items():
        if frame_idx >= num_frames:
            continue
        frame_poses = video_poses[frame_idx]

        for bbox_idx, roi_bbox in enumerate(bboxes):
            x1, y1 = roi_bbox[:2]
            x2 = x1 + roi_bbox[2]
            y2 = y1 + roi_bbox[3]
            if dilate:
                x1 -= 5
                y1 -= 10
                x2 += 5
                y2 += 10
            is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
            
            for body_idx in range(num_bodies):
                pose_coords = frame_poses[body_idx]['coords']
                
                num_valid = np.sum(np.all(pose_coords, axis=1))
                if num_valid > 0: # A joints minimum can be set here
                    n_hits = np.apply_along_axis(is_inside, 1, pose_coords).sum()
                    hits_perc = n_hits/num_valid
                    if hits_perc > perc_thres:
                        bodies_hits[body_idx, bbox_idx] += 1
                        body_bbox = compute_bbox(pose_coords)
                        area_inter = compute_area_intersection(roi_bbox,
                                                               body_bbox)
                        bodies_area_inter[body_idx, bbox_idx] += area_inter
    
    hits_thres = len(event_bboxes)*.25
    if debug:
        print("bodies_hits:", bodies_hits)
        print("bodies_area_inter:", bodies_area_inter)
        print("hits_thres:", hits_thres)
    
    sorted_args = np.argsort(bodies_area_inter, axis=None)[::-1]
    bbox_ids_matched = []
    skel_ids_matched = []
    for flat_arg in sorted_args:
        skel_id, bbox_id = np.unravel_index(flat_arg, bodies_area_inter.shape)
        # print(skel_id, bbox_id, bodies_area_inter[skel_id, bbox_id],
        #       bodies_hits[skel_id, bbox_id])
        if (skel_id not in skel_ids_matched and bbox_id not in bbox_ids_matched
            and  bodies_hits[skel_id, bbox_id] >= hits_thres):
            skel_ids_matched.append(skel_id)
            bbox_ids_matched.append(bbox_id)
            # print("> Added!")
        if len(skel_ids_matched) == num_bboxes:
            break
    
    matches = dict(zip(bbox_ids_matched, skel_ids_matched))
    
    bodies_to_keep = [ (matches[bbox_idx] if bbox_idx in matches else None)
                      for bbox_idx in range(num_bboxes)]
    
    # bodies_to_keep = np.asarray([ body_idx 
    #                          for body_idx, body_hits in enumerate(bodies_hits)
    #                          if body_hits >= hits_thres])
    
    if debug:
        print("After matching...")
        print("bodies_to_keep:", bodies_to_keep)
        print("bodies_to_keep:")
        for bbox_idx, body_idx in enumerate(bodies_to_keep):
            if body_idx is not None:
                print(bbox_idx, body_idx, bodies_area_inter[body_idx, bbox_idx],
                      bodies_hits[body_idx, bbox_idx])
            else:
                print(bbox_idx, body_idx)
        # x = [ [bbox_idx, body_idx, bodies_area_inter[body_idx, bbox_idx]]  
        #  for bbox_idx, body_idx in enumerate(bodies_to_keep) ]
        # print("bodies_to_keep:", x)
    
    pruned_poses = [ [ (frame_poses[body_idx] if body_idx is not None else
                        {'coords': np.zeros_like(pose_coords)})
                      for body_idx in bodies_to_keep]
                    for frame_poses in video_poses]
    
    return pruned_poses

def prune_bodies(video_poses, video_gt, metric='central_bboxes'):
    # if video_poses == [] or np.asarray(video_poses).size == 0:
    # if video_poses == [] or not any(video_poses):
    if ((isinstance(video_poses, list) and not any(video_poses)) or 
            (isinstance(video_poses, np.ndarray) and video_poses.size == 0)):
        return video_poses
    pruned_video_poses = None
    
    if metric == 'central_bboxes':
        persons_info = get_persons_info(video_gt)
        bboxes = persons_info[['x','y','width','height']].values
        pruned_video_poses = prune_bodies_central_bboxes(video_poses, bboxes)
    elif metric == 'event_bboxes':
        pruned_video_poses = prune_bodies_event_bboxes(video_poses,
                                                 video_gt.event_bboxes)
    elif metric == 'match_event_bboxes':
        pruned_video_poses = match_bodies_event_bboxes(video_poses,
                                                 video_gt.event_bboxes)
    elif metric == 'event_rois':
        pruned_video_poses = prune_bodies_event_rois(video_poses,
                                               video_gt.event_rois)
    else:
        raise ValueError("Invalid prune metric:"+metric)
    
    return pruned_video_poses

def compute_bbox(skeleton, empty_val=None):
    skel = np.asarray(skeleton)
    x_coords = skel[:,0]
    y_coords = skel[:,1]
    if np.any(x_coords) and np.any(y_coords):
        x = int(np.min(x_coords[np.nonzero(x_coords)]))
        y = int(np.min(y_coords[np.nonzero(y_coords)]))
        w = int(np.max(x_coords[np.nonzero(x_coords)])) - x
        h = int(np.max(y_coords[np.nonzero(y_coords)])) - y
        bbox = [x,y,w,h]
    else:
        bbox = empty_val
    
    return bbox

def compute_area_intersection(bbox1, bbox2):
    x1, y1, w, h = bbox1
    x2 = x1 + w
    y2 = y1 + h
    
    _x1, _y1, _w, _h = bbox2
    _x2 = _x1 + _w
    _y2 = _y1 + _h

    dx = min(x2, _x2) - max(x1, _x1)
    dy = min(y2, _y2) - max(y1, _y1)
    
    if (dx>0) and (dy>0):
        area_inter = dx*dy
    else:
        # area_inter = np.nan
        area_inter = 0
    
    return area_inter

def compute_metric(prvs_coords, curr_coords, metric, verbose=False, 
                   dilate=True, erode=False):
    metrics = []
    for prvs in prvs_coords:
        non_zero_idx_prvs = np.all(prvs, axis=1)
        if not np.any(non_zero_idx_prvs): # all zeroes, no need to compute
            metrics.append([np.nan]*len(curr_coords))
            continue
        
        if metric == 'dist_joints':
            coords_dists = []
            for curr in curr_coords:
                non_zero_idx_curr = np.all(curr, axis=1)
                non_zero_idx = np.logical_and(non_zero_idx_prvs, non_zero_idx_curr)
                if np.any(non_zero_idx): # At least one valid
                    dists = np.linalg.norm(curr - prvs, axis=1)[non_zero_idx]
                    coords_dists.append(np.mean(dists))
                else:
                    # coords_dists.append(9999)
                    coords_dists.append(np.nan)
            metrics.append(coords_dists)
        elif metric == 'dist_center':
            center = prvs[non_zero_idx_prvs].T.mean(axis=-1)
            coords_dists = []
            for curr in curr_coords:
                non_zero_idx_curr = np.all(curr, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    curr_center = curr[non_zero_idx_curr].T.mean(axis=-1)
                # non_zero_idx = np.logical_and(non_zero_idx_prvs, non_zero_idx_curr)
                if np.any(non_zero_idx_prvs) and np.any(non_zero_idx_curr): 
                    # At least one valid
                    dist = np.linalg.norm(curr_center - center)
                    coords_dists.append(dist)
                else:
                    # coords_dists.append(9999)
                    coords_dists.append(np.nan)
            metrics.append(coords_dists)
        elif metric == 'num_joints':
            bbox = compute_bbox(prvs)
            if dilate:
                bbox = transform_bbox(bbox, ratio=1.1)
            if erode:
                bbox = transform_bbox(bbox, ratio=0.9)
            # x1, y1 = bbox[:2]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            is_inside = lambda p: x1 < p[0] < x2 and y1 < p[1] < y2
            misses_perc = []
            for curr in curr_coords:
                num_valid = np.sum(np.all(curr, axis=1))
                if num_valid == 0:
                    misses_perc.append(np.nan)
                else:
                    n_hits = np.apply_along_axis(is_inside,1,curr).sum()
                    precision = n_hits/num_valid
                    recall = n_hits/len(curr)
                    f1 = 2*precision*recall/(precision+recall)
                    # if verbose: print(precision, recall, f1, 1-precision)
                    # misses_perc.append(1 - precision)
                    misses_perc.append(1 - f1)
            metrics.append(misses_perc)
        elif metric == 'area_intersection':
            bbox = compute_bbox(prvs)
            if dilate:
                # bbox = transform_bbox(bbox, ratio=1.2)
                bbox = transform_bbox(bbox, abs_values=(10,10))
            if erode:
                bbox = transform_bbox(bbox, ratio=0.8)
            # x1, y1 = bbox[:2]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            areas = []
            for curr in curr_coords:
                curr_bbox = compute_bbox(curr)
                
                if curr_bbox is not None:
                    area_inter = compute_area_intersection(bbox, curr_bbox)
                else:
                    area_inter = 0
                
                # if not np.isnan(area_inter):
                if area_inter > 0:
                    areas.append(1e+3/area_inter) 
                    # Inverting to make smaller = better, keeping consistent w/
                    # the other metrics.
                else:
                    areas.append(np.nan)
                
                # if curr_bbox is not None:
                #     c_x1, c_y1, c_w, c_h = curr_bbox
                #     c_x2 = c_x1 + c_w
                #     c_y2 = c_y1 + c_h
                
                #     dx = min(x2, c_x2) - max(x1, c_x1)
                #     dy = min(y2, c_y2) - max(y1, c_y1)
                #     # if (dx>=0) and (dy>=0):
                #     if (dx>0) and (dy>0):
                #         area_inter = dx*dy
                #         areas.append(1e+3/area_inter)
                #     else:
                #         # areas.append(1e+3)
                #         areas.append(np.nan)
                # else:
                #     areas.append(np.nan)
            
            metrics.append(areas)
        else:
            raise ValueError("Invalid metric choice: " + metric)
    return metrics

def track_bodies(video_poses, extrapolate=True, metric='dist_joints', 
                 interpolate=True, metric_th=None, last_k=10, first_k_valid=2,
                 keep_most_common=False, min_freq_perc=.1, max_num_ppl=None,
                 check_for_zeros=False, keep_top_k=None, debug=False,
                 cam_motions=None):
    """
    Track the poses through all frames, attributing the skeletons to the most 
    likely body, or create a new body if none is close enough.
    This is necessary because the pose estimation is done by frame, there is 
    no consistency between frames, i.e. throughout the video.
    Change log:
        V1: 
            - track based on distances of non-zero joints
            - if no skeleton is found for body, repeat last valid + camera motion
            - use camera motion info to extrapolate missing skeletons in the begining
    (current) V2:
            - normalize skeletons coords: (apply_center_normalization?)
                a) use camera motion per frame to translate coords by frame
                b) use central position during all video as new center

    Parameters
    ----------
    video_poses : list
        Estimated poses for all frames.
    extrapolate : bool, optional
        Generate pose for missing estimations before and after detection.
        The default is True.
    interpolate : bool, optional
        Generate pose for missing estimations between detections
        The default is True.
    metric : string, optional
        How to calculate the metric used for tracking the bodies. 
        Available optiosn are: 
            'dist_center', distance to the center
            'dist_joints', distance between the joints
            'num_joints', percentage of joints inside prvs joints bbox
            'area_intersection', area of intersection between joints bbox
        The default is 'dist_joints'.

    Returns
    -------
    tracked_video_poses : list
        Poses for all frames, with an attempted body consistency for the video.

    """
    # if video_poses == [] or np.asarray(video_poses).size == 0:
    # if video_poses == [] or not any(video_poses):
    if ((isinstance(video_poses, list) and not any(video_poses)) or 
            (isinstance(video_poses, np.ndarray) and video_poses.size == 0)):
        return video_poses
    
    if cam_motions is not None:
        video_poses = apply_cam_motion(video_poses, cam_motions, 'coords', 
                                       poses=True)
    
    for frame_poses in video_poses:
        if frame_poses == []:
            continue
        else:
            pose = frame_poses[0]
            num_joints = len(pose['coords'])
            break
    
    if check_for_zeros:
        video_poses = [ [ o_p for o_p in frame_poses if np.all(o_p['coords']) ]
                       for frame_poses in video_poses ]
    
    # print("Total number of detections:", np.sum([len(f) for f in video_poses]))
    
    # null_joints = np.array([ [-1, -1] for _ in range(num_joints) ])
    null_joints = np.array([ [0, 0] for _ in range(num_joints) ])
    if max_num_ppl is None:
        max_num_ppl = np.max([ len(frame_poses) for frame_poses in video_poses])
        max_num_ppl *= 2 # Adding space for more ppl just in case
    bodies_coords = [ [] for _ in range(max_num_ppl) ]
    pose_idx = -1
    for pose_idx, pose in enumerate(video_poses[0]):
        # bodies_coords[pose_idx].append(pose['coords'].tolist())
        bodies_coords[pose_idx].append(pose['coords'].copy())
    # for missing_idx in range(pose_idx+1,max_num_ppl):
    for missing_idx in range(pose_idx+1,max_num_ppl):
        bodies_coords[missing_idx].append(null_joints.copy())
    null_bodies_coords_idx = (~np.any(bodies_coords, axis=(1,2,3)))
    
    prvs_coords = [ np.array(body[0]) for body in bodies_coords ]
    prvs_coords_log = []
    all_cam_motions = ( cam_motions if cam_motions is not None else [] )
    ### Main loop - computing metrics
    for prvs_idx, frame_poses in enumerate(video_poses[1:]):
        ### Setting frame idx to investigate
        verbose = debug and (prvs_idx in [])
        # verbose = True
        if verbose: print("+++ prvs_idx:", prvs_idx)
        if frame_poses == []: # Skipping frame because there is no pose
            for body_coords in bodies_coords:
                body_coords.append(null_joints.copy())
                # body_coords.append(body_coords[-1])
            if cam_motions is None:
                all_cam_motions.append([0,0])
            # Was not updating prvs_coords_log or prvs_coords before...
            prvs_coords = [ np.array(body[-1]) for body in bodies_coords ]
            prvs_coords_log.append([ body_prvs_coords.copy() for body_prvs_coords 
                                in prvs_coords])
            continue
        
        curr_coords = [ person['coords'].copy() for person in frame_poses ]
                
        ### Computing metric
        joints_distances = compute_metric(prvs_coords, curr_coords, metric, verbose)
            
        distances = np.array(joints_distances)
        # print(distances[:,0])
        
        ### Average current distance with prvs distance from prvs_coords_log
        ## Loop per previous log frames
        # dists_list = [distances]
        # if verbose: print(np.asarray(distances).shape)
        # # prvs_coords_log -> (n_frames, n_bodies, n_joints, n_dim)
        # for older_prvs_coords in prvs_coords_log[-last_k:]:
        #     joints_distances = compute_metric(older_prvs_coords, curr_coords, metric)
        #     dists_list.append(joints_distances)
        
        
        ## Loop per body
        dists_list = [distances]
        if len(prvs_coords_log) > 0:
            # Reshaping to (n_bodies, n_frames, n_joints, n_dim)
            top_valid_coords = []
            for body_prvs_coords in np.transpose(prvs_coords_log[-last_k:], (1,0,2,3)):
                valid_coords_idx = np.any(body_prvs_coords, axis=(1,2))
                top_valid_body_coords = body_prvs_coords[valid_coords_idx][
                    -first_k_valid:]
                top_valid_body_coords = top_valid_body_coords[::-1]
                missing_top = first_k_valid - len(top_valid_body_coords)
                if missing_top > 0:
                    top_valid_body_coords = np.pad(top_valid_body_coords, 
                                          ((0,missing_top), (0,0), (0,0) ) )
                top_valid_coords.append(top_valid_body_coords)
            for log_idx in range(first_k_valid):
                older_prvs_coords = [ b[log_idx] for b in top_valid_coords ]
                joints_distances = compute_metric(older_prvs_coords, curr_coords, metric)
                dists_list.append(joints_distances)
                
        
        
        ## dists_list.shape (first_k_valid+1, n_bodies, n_coords)
        # if verbose: print(np.asarray(prvs_coords)[[3,8]])
        # if verbose: print(np.asarray(dists_list)[:,[3,8]].T)
        if verbose: print("+++++ dists_list +++++")
        if verbose: print(np.asarray(dists_list)[:,5].T)
        if verbose: print("+++++ prvs_coords_log +++++")
        if verbose: print(np.asarray(prvs_coords_log)[-last_k:,5].T)
        if verbose: print(np.asarray(prvs_coords_log).shape)
        
        with warnings.catch_warnings():
            # Suppressing expected warnings:
            # "RuntimeWarning: Mean of empty slice"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if metric == 'dist_joints':
                distances = np.nanmean(dists_list, axis=0)
            elif metric == 'dist_center':
                distances = np.nanmean(dists_list, axis=0)
            elif metric == 'num_joints':
                distances = np.nanmean(dists_list, axis=0)
                # distances = np.nanmin(dists_list, axis=0)
            elif metric == 'area_intersection':
                distances = np.nanmean(dists_list, axis=0)
                # distances = np.nansum(dists_list, axis=0)
            else:
                distances = np.nanmean(dists_list, axis=0)
        
        if verbose: print(np.asarray(distances)[[3,8]]) # distances.shape (n_bodies,n_coords)
        # if verbose: print(np.asarray(distances)) # distances.shape (n_bodies,n_coords)
        
        ### Threshold for min/max metric value
        if metric_th is None:
            if metric == 'dist_joints': # Hard-coded threshold
                metric_th = 50
                # metric_th = 25
            elif metric == 'dist_center':
                metric_th = 50
            elif metric == 'num_joints':
                metric_th = .75
            elif metric == 'area_intersection':
                metric_th = 500
            
        ### Attribute pose to min distance body, if body still have not been appended yet
        sorted_idx = np.dstack(np.unravel_index(np.argsort(
            distances.ravel()), distances.shape))[0]
        poses_used = []
        bodies_appended = []
        for dist_idx in sorted_idx:
            body_idx, pose_idx = dist_idx
            if body_idx in bodies_appended or pose_idx in poses_used:
                continue
            min_dist = distances[tuple(dist_idx)]
            # if verbose: print(body_idx, pose_idx, min_dist)
            
            if min_dist < metric_th:
            # if min_dist < metric_th or not np.any(null_bodies_coords_idx):
                bodies_coords[body_idx].append(curr_coords[pose_idx])
                poses_used.append(pose_idx)
                bodies_appended.append(body_idx)
                # print(pose_idx, body_idx, min_dist)
            # else: # Append skel to null_joint body
            elif np.any(null_bodies_coords_idx):
                body_idx = np.argwhere(null_bodies_coords_idx)[0,0]
                null_bodies_coords_idx[body_idx] = False
                bodies_coords[body_idx].append(curr_coords[pose_idx])
                poses_used.append(pose_idx)
                bodies_appended.append(body_idx)
            # else:
            #     break # No other pair will be valid
                
                # print(body_idx, pose_idx, min_dist)
            #     print('curr_coords[pose_idx]', curr_coords[pose_idx][:2])
            if len(poses_used) == len(curr_coords):
                break
        # print(np.array([bodies_appended, poses_used]).T)
        
        
        if cam_motions is None:
            ### Calculating camera motion
            motions = []
            for body_idx in bodies_appended:
                body = bodies_coords[body_idx]
                motion = body[-1] - body[-2]
                non_zero_idx = np.logical_and(np.all(body[-1], axis=1), 
                                              np.all(body[-2], axis=1))
                # motions.append(motion[non_zero_idx])
                motions += motion[non_zero_idx].tolist()
            if motions != []:
                motions = np.asarray(motions)
                cam_motion = motions.mean(axis=0)
            else:
                cam_motion = [0,0]
            all_cam_motions.append(cam_motion)
        
        ### Making sure all bodies are appended
        for body_coords in bodies_coords:
            if len(body_coords) == (prvs_idx+1):
                ## Use null_joints (zeroes)
                body_coords.append(null_joints.copy())
                
                ## Use previous coords as it was
                # body_coords.append(body_coords[-1])
                
                ## Use previous coords plux camera motion compensation
                # last_coords = body_coords[-1].copy()
                # non_zero = np.all(last_coords, axis=1)
                # last_coords[non_zero] = last_coords[non_zero] + cam_motion
                # body_coords.append(last_coords)
        
        
        ### Updating prvs_coords_log (memory)
        
        ## Applying camera motion compensation to prvs coords
        if cam_motions is None:
            for older_prvs_coords in prvs_coords_log[-last_k+1:]:
                for body_prvs_coords in older_prvs_coords:
                    non_zero = np.all(body_prvs_coords, axis=1)
                    adjusted_coords = body_prvs_coords[non_zero] + cam_motion
                    body_prvs_coords[non_zero] = adjusted_coords
        
        # prvs_coords_log.append([np.array(body[-2]) for body in bodies_coords])
        prvs_coords_log.append([ body_prvs_coords.copy() for body_prvs_coords 
                                in prvs_coords])
        # print("> prvs_coords_log:", np.array(prvs_coords_log)[-1,:,0])
        
        prvs_coords = [ np.array(body[-1]) for body in bodies_coords ]
        # Updating prvs_coords for the non-zero values
        # for body_coords, body_prvs_coords in zip(bodies_coords, prvs_coords):
        #     last_body_coords = body_coords[-1]
        #     body_prvs_coords[last_body_coords > 0] = last_body_coords[last_body_coords > 0]
        
        ### Setting limit of runs
        # if prvs_idx == 2:
            # break
    
    # Remove null_bodies
    if np.any(null_bodies_coords_idx):
        first_null_body_idx = np.argwhere(null_bodies_coords_idx)[0,0]
        bodies_coords = bodies_coords[:first_null_body_idx]
    
    # Checking number of valid poses before interpolate and extrapolate
    num_valid_poses = np.array([ np.sum([ np.any(pose) for pose in body_coords ]) 
                       for body_coords in bodies_coords ])
    
    if debug:
        print(np.argsort(num_valid_poses)[::-1])
        print(np.sort(num_valid_poses)[::-1])
        print("Number of poses:", len(num_valid_poses))
        print("Min number of frames:", len(bodies_coords[0])*min_freq_perc )
    
    ### Drop bodies with less than 10% of valid poses
    # print(len(bodies_coords))
    condition = (num_valid_poses >= len(bodies_coords[0])*min_freq_perc)
    
    if keep_most_common:
        condition2 = (num_valid_poses == np.max(num_valid_poses))
        condition = np.logical_and(condition, condition2)
    
    if keep_top_k is not None:
        cut_value = np.sort(num_valid_poses)[-keep_top_k:][0]
        condition2 = (num_valid_poses >= cut_value)
        condition = np.logical_and(condition, condition2)
    
    bodies_to_keep = np.argwhere(condition).flat
    if keep_top_k is not None:
        bodies_to_keep = bodies_to_keep[:keep_top_k] # avoid multiple valids
        # Sort by num_valid, keep more common first
        sorted_idx = num_valid_poses[bodies_to_keep].argsort()[::-1]
        bodies_to_keep = bodies_to_keep[sorted_idx]
    bodies_coords = [ bodies_coords[body_idx] for body_idx in bodies_to_keep ]
    
    if interpolate and bodies_coords != []:
        num_bodies = len(bodies_coords)
        num_frames = len(bodies_coords[0])
        num_joints = len(bodies_coords[0][0])
        num_dim = len(bodies_coords[0][0][0])
        all_frames_idx = np.arange(num_frames)
        # print("Warning: only running for one player!")
        # for body_idx in [12]:
        for body_idx in range(num_bodies):
            # if num_valid_poses[body_idx] == 1: # can not interpolate with 1 value
            if num_valid_poses[bodies_to_keep[body_idx]] == 1:
                # Not possible to interpolate with 1 value only
                continue
            
            body_coords = bodies_coords[body_idx]
            
            new_body_coords = []
            ## Loop over joint
            # print("Warning: only running for one joint!")
            # for joint_idx in [15]:
            for joint_idx in range(num_joints):
                joint_coords = [ pose[joint_idx] for pose in body_coords ]
                gt_frames_mask = np.any(joint_coords, axis=1)
                gt_frames_idx = all_frames_idx[gt_frames_mask]
                
                if gt_frames_idx.size > 1:
                    # Do not interpolate if gap is too big
                    gaps = np.diff(gt_frames_idx)
                    if np.any(gaps > 5):
                        for idx in np.argwhere(gaps > 5).flat:
                            frame_start = gt_frames_idx[idx] +1
                            frame_end = gt_frames_idx[idx+1]
                            gt_frames_mask[frame_start:frame_end] = True
                        gt_frames_idx = all_frames_idx[gt_frames_mask]
                    
                    valid_joint_coords = [ joint_coords[i] for i in gt_frames_idx ]
                    
                    new_joint_coords = []
                    for dim_idx in range(num_dim):
                        fp = [ c[dim_idx] for c in valid_joint_coords ]
                        interp = np.interp(all_frames_idx, gt_frames_idx, fp, 
                                           left=0, right=0)
                        new_joint_coords.append(interp)
                    new_joint_coords = np.asarray(new_joint_coords).T
                    
                    # fp = [ c[1] for c in valid_joint_coords ]
                    # all_y = np.interp(all_frames_idx,gt_frames_idx, fp)
                else:
                    # new_joint_coords = np.zeros((num_frames, num_dim))
                    new_joint_coords = np.array(joint_coords)
                new_body_coords.append(new_joint_coords)
            new_body_coords = np.transpose(new_body_coords, (1,0,2))
            bodies_coords[body_idx] = new_body_coords
    
    if extrapolate and bodies_coords != []:
        num_bodies = len(bodies_coords)
        num_frames = len(bodies_coords[0])
        num_joints = len(bodies_coords[0][0])
        # num_dim = len(bodies_coords[0][0][0])
        for body_idx in range(num_bodies):
        # for body_idx in [3]:
        # for body_idx in [6]:
            # print("> body_idx:", body_idx)
            body_coords = bodies_coords[body_idx]
            valid_frames_mask = [ np.any(f) for f in body_coords ]
            first_valid = np.argmax(valid_frames_mask)
            last_valid = num_frames - 1 - np.argmax(valid_frames_mask[::-1])
            # print(first_valid, last_valid)
            
            prvs_coords = body_coords[first_valid]
            for frame_idx in reversed(range(first_valid)):
                curr_coords = body_coords[frame_idx]
                cam_motion = all_cam_motions[frame_idx]
                for jnt_idx in range(num_joints):
                    curr_j = curr_coords[jnt_idx]
                    prvs_j = prvs_coords[jnt_idx]
                    if not np.any(curr_j) and np.any(prvs_j):
                        new_coords = prvs_j - cam_motion
                        bodies_coords[body_idx][frame_idx][jnt_idx] =new_coords
                prvs_coords = bodies_coords[body_idx][frame_idx]
            
            prvs_coords = body_coords[last_valid]
            for frame_idx in range(last_valid+1, num_frames):
                # print("> > frame_idx:", frame_idx)
                curr_coords = body_coords[frame_idx]
                # print(np.asarray(bodies_coords[body_idx][frame_idx]).T)
                cam_motion = all_cam_motions[frame_idx-1]
                for jnt_idx in range(num_joints):
                    curr_j = curr_coords[jnt_idx]
                    prvs_j = prvs_coords[jnt_idx]
                    if not np.any(curr_j) and np.any(prvs_j):
                        new_coords = prvs_j + cam_motion
                        bodies_coords[body_idx][frame_idx][jnt_idx] =new_coords
                # print(np.asarray(bodies_coords[body_idx][frame_idx]).T)
                prvs_coords = bodies_coords[body_idx][frame_idx]
            
    # print(np.array(bodies_coords[10]).transpose((0,2,1)))
    # print(np.array(bodies_coords)[:,0,:,0])
    
    # print(np.asarray(all_cam_motions))
    
    tracked_video_poses = []
    for frame_idx in range(len(video_poses)):
        tracked_video_poses.append([ 
            {'coords': np.array(body_coords[frame_idx])} 
            for body_coords in bodies_coords ])
    
    if cam_motions is not None:
        tracked_video_poses = apply_cam_motion(tracked_video_poses, 
                                           -cam_motions, 'coords', poses=True)
    
    return tracked_video_poses

def complement_bodies(video_bodies, all_video_poses, poses=True):
    """
    Greedily complement the bodies (tracked poses) with the most suitable poses.
    Scan poses on frames before/after tracked pose starts/ends and check if
    any of the poses matches, then appends the best match to the tracked body.

    Parameters
    ----------
    video_bodies : TYPE
        DESCRIPTION.
    all_video_poses : TYPE
        DESCRIPTION.
    poses : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if poses:
        bodies_coords = np.array(video_poses_to_coords(video_bodies))
        all_video_coords = video_poses_to_coords(all_video_poses)
    else:
        bodies_coords = np.array(video_bodies)
        all_video_coords = np.array(all_video_poses)
    
    if bodies_coords.size == 0:
        return video_bodies
    
    num_frames = bodies_coords.shape[0]
    
    for body_idx in range(bodies_coords.shape[1]):
        # print("> body_idx:", body_idx)
        body_coords = bodies_coords[:,body_idx]
        
        ## Complementing in the begining
        pose_appended = True
        while pose_appended:
        # for _ in range(1):
            pose_appended = False
            
            body_appearances = np.any(body_coords, axis=(1,2))
            # print(body_coords[:,0])
            first_frame_idx = body_appearances.argmax()
            # print(first_frame_idx)
            
            if first_frame_idx == 0:
                break
            
            body_frame_coords = body_coords[first_frame_idx]
            lookup_frame_idx = first_frame_idx-1
            
            frame_coords = all_video_coords[lookup_frame_idx]
            
            if frame_coords == []:
                break
            
            coords_metrics = compute_metric([body_frame_coords], frame_coords, 
                                   metric='area_intersection')[0]

            if np.isnan(coords_metrics).all():
                break
            
            min_coords_idx = np.nanargmin(coords_metrics)
            min_metric = coords_metrics[min_coords_idx]
            
            # print(body_frame_coords)
            # print(frame_coords)
            # print(coords_metrics)
            # print(min_metric)
            # print(min_coords_idx)
            
            metric_th = 999
            if min_metric < metric_th:
                body_coords[lookup_frame_idx] = frame_coords[min_coords_idx]
                pose_appended = True
        
        ## Complementing in the ending
        pose_appended = True
        while pose_appended:
        # for _ in range(3):
            pose_appended = False
            
            body_appearances = np.any(body_coords, axis=(1,2))
            last_frame_idx = num_frames-1 - np.flip(body_appearances).argmax()
            # print("last_frame_idx:", last_frame_idx)
            
            if last_frame_idx == num_frames-1:
                break
            
            body_frame_coords = body_coords[last_frame_idx]
            lookup_frame_idx = last_frame_idx+1
            
            frame_coords = all_video_coords[lookup_frame_idx]
            
            if frame_coords == []:
                break
            
            coords_metrics = compute_metric([body_frame_coords], 
                                   frame_coords, 
                                   metric='area_intersection')[0]

            if np.isnan(coords_metrics).all():
                break
            
            min_coords_idx = np.nanargmin(coords_metrics)
            min_metric = coords_metrics[min_coords_idx]
            
            # print(body_frame_coords)
            # print(frame_coords)
            # print(coords_metrics)
            # print(min_metric)
            # print(min_coords_idx)
            
            metric_th = 999
            if min_metric < metric_th:
                body_coords[lookup_frame_idx] = frame_coords[min_coords_idx]
                pose_appended = True
                # print("appending!")
            
        bodies_coords[:,body_idx] = body_coords
    
    if poses:
        bodies_coords = np.asarray(video_coords_to_poses(bodies_coords))
    
    return bodies_coords

def merge_bodies(video_bodies, poses=True, dist_th=100, search_window=3):
    if poses:
        bodies_coords = np.asarray(video_poses_to_coords(video_bodies))
    else:
        bodies_coords = np.asarray(video_bodies)
    
    if bodies_coords.size == 0:
        return video_bodies
    
    num_frames = bodies_coords.shape[0]
    bodies_coords = np.transpose(bodies_coords, (1,0,2,3))
    
    # Sorting by order of appearance
    bodies_appearance = np.all(bodies_coords, axis=(2,3))
    bodies_first_appearance = bodies_appearance.argmax(axis=1)
    
    bodies_coords = bodies_coords[np.argsort(bodies_first_appearance)]
    bodies_appearance = bodies_appearance[np.argsort(bodies_first_appearance)]
    
    merged_bodies = []
    merged_bodies_idx = []
    for body_idx, body in enumerate(bodies_coords):
    # for body_idx, body in enumerate(bodies_coords[:1]):
        # print("> body_idx", body_idx)
        if body_idx in merged_bodies_idx:
            ### TO DO should I remove or not the body already merged?
            # I can remove because these body coords will now be present at the other one?
            # Problem case is if both have long occurs, but only near end they get close
            # Should I limit the merge only to those that recently appeared?
            continue
        merged_body = np.array(body)
        
        if body_idx == len(bodies_coords)-1: # Last
            merged_bodies.append(merged_body)
            continue
        
        curr_merged_bodies_idx = []
        
        merge_occurred = True
        while merge_occurred:
            merge_occurred = False
            
            body_appearances = np.all(merged_body, axis=(1,2))
            last_frame_idx = num_frames-1 - np.flip(body_appearances).argmax()
            
            if last_frame_idx == num_frames-1:
                break
            
            ## Checking around last_frame
            window_dists = []
            for frame_idx in reversed(range(last_frame_idx-search_window, 
                                            last_frame_idx+1)):
                body_frame_coords = merged_body[frame_idx]
                other_bodies_frame_coords = [ other_body[frame_idx]
                                  for other_body in bodies_coords[body_idx+1:]]
                
                dists = compute_metric([body_frame_coords], 
                                       other_bodies_frame_coords, 
                                       metric='dist_center')[0]
                
                # Prevent merging to the same body again
                for other_body_idx in curr_merged_bodies_idx:
                    dists[other_body_idx] = np.nan
                
                window_dists.append(dists)
            # print(np.asarray(window_dists).T)
            
            if window_dists == [] or np.isnan(window_dists).all():
                break
            
            with warnings.catch_warnings():
                # Suppressing expected warnings:
                # "RuntimeWarning: Mean of empty slice"
                warnings.simplefilter("ignore", category=RuntimeWarning)
                min_frame = np.nanargmin(np.nanmin(window_dists, axis=1))
                min_body = np.nanargmin(window_dists[min_frame])
            min_dist = window_dists[min_frame][min_body]
            
            # print(min_frame, min_body, min_dist)
            if min_dist < dist_th:
                merge_frame_idx = last_frame_idx - min_frame
                other_body_idx = body_idx + min_body + 1
                # print(merge_frame_idx, other_body_idx)
                
                merged_bodies_idx.append(other_body_idx)
                curr_merged_bodies_idx.append(min_body)
                
                other_body = bodies_coords[other_body_idx]
                merged_body[merge_frame_idx+1:] = other_body[merge_frame_idx+1:]
                
                ## Smooth region around merging frame
                # Weights are used to prevent averaging zeros
                merged_body[merge_frame_idx] = np.average(
                    merged_body[[merge_frame_idx-1, merge_frame_idx+1]],
                    weights=(merged_body[[merge_frame_idx-1, merge_frame_idx+1]] != 0),
                    axis=0)
                # smoothing_window = 3
                # smoothing_slice = slice(merge_frame_idx-smoothing_window+1,
                #                         merge_frame_idx+1)
                # smoothed_coords = np.average([ merged_body[smoothing_slice],
                #                                other_body[smoothing_slice]], 
                #                              axis=0)
                # merged_body[smoothing_slice] = smoothed_coords
                
                merge_occurred = True
        
        merged_bodies.append(merged_body)
    
    merged_bodies = np.transpose(merged_bodies, (1,0,2,3))
    if poses:
        merged_bodies = video_coords_to_poses(merged_bodies)
    
    return merged_bodies

def read_video_poses(video_gt, pose_style='OpenPose', normalization=None, 
         track=True, track_metric='dist_joints', extrapolate=True, 
         interpolate=True, prune=True, prune_metric='central_bboxes',
         poses_slice=None, filter_rois=False, timesteps=None, conf_thres=0.05,
         hdf_filepath=None, per_player=False):
    # if pose_style == 'OpenPose':
    if pose_style in ['OpenPose','AlphaPose']:
        video_keypoints_dir = video_gt.path
        # video_keypoints_dir = 'D:\\PhD\\volleyball\\skeletons_subsample\\0\\3596'
        
        if pose_style == 'AlphaPose':
            video_keypoints_dir = video_keypoints_dir.replace('skeletons',
                                                              'alphapose')
        
        prefix, clip = os.path.split(video_gt.path)
        prefix, video = os.path.split(prefix)
        prefix, file = os.path.split(prefix)
        if hdf_filepath is None:
            hdf_filepath = os.path.join(prefix, 'raw_skeletons.hdf5')
        
        # if not os.path.exists(hdf_filepath):
        if True: # Reading from jsons files
            if os.path.isdir(video_keypoints_dir): # Volleyball
                json_list = glob.glob(video_keypoints_dir+'/*.json')
                json_list.sort(key=lambda p: 
                           int(os.path.basename(p)[:-15]).replace('frame',''))
                # json_list.sort()
            else: # CAD and CAD-New
                dirpath = os.path.dirname(video_keypoints_dir)
                json_list = glob.glob(dirpath+'/*.json')
                json_list.sort(key=lambda p: 
                           int(os.path.basename(p)[:-15].replace('frame','')) )
                
                # ts = (timesteps if timesteps else 10) #
                ts = (timesteps if timesteps else 20)
                mid_frame_idx = video_gt.frame-1
                ## trim begin if negative
                # clip_slice = slice(mid_frame_idx - int(np.ceil((ts-1)/2)),
                clip_slice = slice(max(0,mid_frame_idx-int(np.ceil((ts-1)/2))),
                                   1 + mid_frame_idx + int(np.floor((ts-1)/2)))
                json_list = json_list[clip_slice]
                
                if len(json_list) < ts: # pad with last skeletons
                    json_list += [json_list[-1]] * (ts-len(json_list))
            
            if json_list == []:
                raise FileNotFoundError("Error reading keypoints at: "+
                                        video_keypoints_dir)
            
            if poses_slice is not None:
                json_list = json_list[poses_slice]
    
            video_poses = []
            for json_file in json_list:
                if pose_style == 'OpenPose':
                    people = parse_json(json_file, prune, conf_thres)
                elif pose_style == 'AlphaPose':
                    people = parse_json_alpha(json_file, prune, conf_thres)
                video_poses.append(people)
        else: # Reading from hdf file
            data_path = '{}/{}'.format(video, clip)
            with h5py.File(hdf_filepath, "r") as f:
                video_grp = f[data_path]
                frames_list = list(video_grp.keys())
                if poses_slice is not None:
                    frames_list = frames_list[poses_slice]
                video_poses = []
                for frame_num in frames_list:
                    frame_coords = video_grp[frame_num]['coords'][()]
                    frame_confs = video_grp[frame_num]['confs'][()]
                    if prune:
                        joints_means = frame_confs.mean(axis=1)
                        frame_coords = frame_coords[joints_means > conf_thres]
                    frame_poses = [ {'coords': np.asarray(pose_coords)} 
                                   for pose_coords in frame_coords ]
                    video_poses.append(frame_poses)
        
        if timesteps is not None and timesteps < len(video_poses):
            offset = (len(video_poses)-timesteps)/2
            central_slice = slice(int(offset), -int(np.ceil(offset)))
            video_poses = video_poses[central_slice]
        
        if not per_player:
            if filter_rois:
                filtered_video_poses = filter_poses_roi(video_poses, video_gt.event_rois)
                video_poses = filtered_video_poses
            
            if track or prune:
                if not track:
                    warnings.warn("Forcing track because prunning requires it.",
                                  UserWarning)
                
                # v3: motions 
                cam_motions = read_preproc_data(video_gt=video_gt,
                      hdf_filename='cam_motions-cv2.hdf5', ignore_no_file=True)
                video_poses = track_bodies(video_poses, metric=track_metric,
                              extrapolate=extrapolate, interpolate=interpolate,
                              cam_motions=cam_motions)
            
            if prune:
                video_poses = prune_bodies(video_poses, video_gt, 
                                                  metric=prune_metric)
            
                # v3: Complement (better than extrapolate?)
                if filter_rois:
                    video_poses = complement_bodies(video_poses, 
                                                    filtered_video_poses)
        else:
            cam_motions = read_preproc_data(video_gt=video_gt,
                                           hdf_filename='cam_motions-cv2.hdf5')
            num_frames = len(video_poses)
            proc_video_poses = [ [] for _ in range(num_frames) ]
            
            event_bboxes = video_gt.event_bboxes
            offset_before = list(event_bboxes.keys())[0]
            last_frame_idx = list(event_bboxes.keys())[-1]
            offset_after = num_frames -last_frame_idx -1
            
            num_players = len(list(event_bboxes.values())[0])
            for p_idx in range(num_players):
                player_event_bboxes = dict(zip(event_bboxes.keys(), 
                      [[players[p_idx]] for players in event_bboxes.values()]))
                
                ## Padding can be dynamic by checking keys() and num_frames
                player_event_rois = [[]]*offset_before + [p[0] for p in 
                           player_event_bboxes.values()] + [[]]*offset_after
                
                filtered_video_poses = filter_poses_roi(video_poses, 
                                            player_event_rois, perc_thres=.75)
                
                player_video_poses = track_bodies(filtered_video_poses, 
                            metric=track_metric, extrapolate=extrapolate,
                            interpolate=interpolate, cam_motions=cam_motions)
                
                player_video_poses = prune_bodies_event_bboxes(player_video_poses,
                                             player_event_bboxes, dilate=False)
                if np.asarray(player_video_poses).size > 0:
                    player_video_poses = complement_bodies(player_video_poses, 
                                                          filtered_video_poses)
                else:
                    # print("Warning: No valid pose found!")
                    # print("video_gt.name, video_gt.video, video_gt.frame, p_idx")
                    # print(video_gt.name, video_gt.video, video_gt.frame, p_idx)
                    player_video_poses = video_coords_to_poses(
                                                np.zeros((num_frames,1,25,2)) )
                
                for frm_idx in range(num_frames):
                    proc_video_poses[frm_idx].append(player_video_poses[frm_idx][0])
            
            video_poses = proc_video_poses
        
        video_poses = video_poses_to_coords(video_poses)
    # elif pose_style == 'OpenPose-preproc':
    elif pose_style in ['OpenPose-preproc','AlphaPose-preproc']:
        if hdf_filepath is None:
            hdf_filename = ('skeletons.hdf5' if pose_style =='OpenPose-preproc'
                            else 'alphapose.hdf5')
        elif '\\' in hdf_filepath or '/' in hdf_filepath:
            hdf_filename = None
        else:
            hdf_filename = hdf_filepath
            hdf_filepath = None
        
        # hdf_filename = ('skeletons.hdf5' if hdf_filepath is None else None)
        video_poses = read_preproc_data(hdf_filepath=hdf_filepath, 
                                  hdf_filename=hdf_filename, video_gt=video_gt)
        
    if normalization is not None:
        video_poses = apply_norm(video_poses, normalization, video_gt)

    return video_poses

def insert_joint_idx(person_joints, num_joints, scale):
    for idx in range(num_joints):
        person_joints[idx].append(idx/scale)
        # person_joints[idx].append(idx/(num_joints-1)) # div_len_idx
        
        # one_hot = np.zeros(num_joints)
        # one_hot[idx] = 1
        # person_joints[idx] += one_hot.tolist()
    pass

def insert_body_part(person_joints, num_joints, scale, body_parts_mapping):
    # num_body_parts = len(np.unique(body_parts_mapping))
    for idx in range(num_joints):
        body_part_idx = body_parts_mapping[idx]/scale
        # body_part_idx = body_parts_mapping[idx]/4 # div_len_idx
        person_joints[idx].append(body_part_idx)
        
        # one_hot = np.zeros(num_body_parts)
        # one_hot[body_parts_mapping[idx]] = 1
        # person_joints[idx] += one_hot.tolist()
    pass

def append_obj(joints_coords, objs_coords, skip_timesteps, n_joints, n_frames):
    if skip_timesteps is not None:
        objs_coords = objs_coords[::skip_timesteps]
    if len(objs_coords[0])< n_joints or len(objs_coords)< n_frames:
        pad_width = ((0, n_frames - len(objs_coords)), 
                     (0, n_joints - len(objs_coords[0])), (0,0))
        objs_coords = np.pad(objs_coords, pad_width=pad_width)
    if joints_coords.size > 0:
        joints_coords = np.concatenate([joints_coords,
                              objs_coords[:,np.newaxis,:,:]], axis=1)
    else:
        joints_coords = objs_coords[:,np.newaxis,:,:]
    
    return joints_coords

def get_Y_data(gt_split, num_classes_indiv=None, num_classes_grp=None, 
               flat_seqs=False, num_indivs=12, indiv_ova_class=None,
               sort_players=False):
    Y_indiv, Y_grp = [], []
    for vid_idx, video_gt in gt_split.iterrows():
        Y_grp.append(video_gt.grp_activity_id)
        
        if num_classes_indiv is not None:
            persons_info = get_persons_info(video_gt, prune_missing=False)
            indiv_actions = persons_info.action_id.values[:num_indivs]
            
            if sort_players:
                ### TODO sort_players
                sorted_idx = argsort_players(video_gt)
                sorted_idx = [ (idx if idx is not None else 11) 
                              for idx in sorted_idx ]
                indiv_actions = indiv_actions[sorted_idx]
            
            Y_indiv += indiv_actions.tolist()
            
            # if flat_seqs:
            #     raise NotImplementedError("get_Y_data with flat_seqs=True")
            #     # Y_indiv += [indiv_actions.tolist()] * len(all_seqs)
            # else:
            #     Y_indiv.append(indiv_actions.tolist())
    
    Y = []
    if num_classes_grp is not None:
        Y_grp = to_categorical(Y_grp, num_classes_grp)
        Y.append(Y_grp)
    if num_classes_indiv is not None:
        Y_indiv = np.asarray(Y_indiv)
        most_common = np.bincount(Y_indiv[Y_indiv!=-1]).argmax()
        Y_indiv[Y_indiv==-1] = most_common
        
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
                  Y_indiv.reshape((-1, len(Y_grp), num_classes_indiv)) ]
    
    if len(Y) == 1:
        Y = Y[0]
    
    return Y

def get_hoop_coords(video_gt, hdf_filepath=None, use_center=False):
    if hdf_filepath is None:
        from . import detector
        # video_frames = read_video(dataset.get_video_filepath(video_gt))
        video_frames = read_video(video_gt.video_path)
        
        # video_hoop_bboxes = []
        # for frame in video_frames:
        #     hoopbBoxes = detector.detect_hoop(frame, verbose=False)
        #     video_hoop_bboxes.append(hoopbBoxes)
        center_bbox = get_video_center_bbox(video_gt, ratio=.75)
        center_bbox[3] = center_bbox[3] - center_bbox[3]//2 # Upper part only
        video_hoop_bboxes = []
        for frame_idx in range(len(video_frames)):
            frame = video_frames[frame_idx]
            if frame_idx < len(video_gt.event_rois):
                event_roi = video_gt.event_rois[frame_idx]
                event_roi = transform_bbox(event_roi, ratio=.75)
            else:
                event_roi = None
            hoopbBoxes = detector.detect_hoop(frame, roi=center_bbox, 
                                              non_roi=event_roi, merge=True)
            video_hoop_bboxes.append(hoopbBoxes)
    
        preproc_hoops_coords = detector.track_and_prune_hoops(video_hoop_bboxes)
        
        if len(preproc_hoops_coords[0]) != 0:
            # preproc_hoops_coords = [ frame_pose[0]['coords']
            #             for frame_pose in preproc_hoops_coords]
            preproc_hoops_coords = [ [pose['coords'] for pose in frame_poses]
                        for frame_poses in preproc_hoops_coords]
        else:
            # No hoop detected
            # preproc_hoops_coords = [ [[0., 0.],[0., 0.]]
            #             for _ in range(len(video_hoop_bboxes))]
            preproc_hoops_coords = [ [[[0., 0.],[0., 0.]]]
                        for _ in range(len(video_hoop_bboxes))]
    else:
        prefix, clip = os.path.split(video_gt.path)
        prefix, video = os.path.split(prefix)
        data_path = video+'/'+clip
        with h5py.File(hdf_filepath, "r") as f:
            preproc_hoops_coords = f[data_path][()]
    
    if use_center:
        centers = np.transpose(preproc_hoops_coords, (0,1,3,2)).mean(axis=-1)
        preproc_hoops_coords = centers
    
    return preproc_hoops_coords

def get_balls_coords(video_gt, hdf_filepath=None, use_center=False, 
             normalization=None, interpolate=True, noise_std=0, noise_dr=0):
    if hdf_filepath is None:
        from . import detector # Local import to avoid unnecessary cv2 import
        video_balls_bboxes = detector.detect_video_balls(video_gt)
        
        ball_det_params = video_gt['ball_det_params'].copy()
        
        center_bbox = get_video_center_bbox(video_gt, ratio=.7, square=True)
        
        pad_top = ball_det_params.pop('pad_top', 0)
        pad_width = ball_det_params.pop('pad_width', 0)
        center_bbox = transform_bbox(center_bbox, pad_top=pad_top)
        center_bbox = transform_bbox(center_bbox, abs_values=(pad_width,0))
        # center_bbox = transform_bbox(center_bbox, ratio=.66, square=True)
        video_rois = [ center_bbox ] * len(video_balls_bboxes)
        
        video_frames = read_video(video_gt.video_path)
        cam_motions = estimate_motion(video_frames)
        
        if video_gt.VideoWidth < 1280:
            preproc_balls_coords = detector.track_and_prune_balls(
                video_balls_bboxes,video_rois=video_rois,cam_motions=cam_motions)
        else:
            preproc_balls_coords = detector.track_and_prune_balls_hd( 
                 video_balls_bboxes,video_rois=video_rois,cam_motions=cam_motions)
        
        if len(preproc_balls_coords[0]) != 0:
            # preproc_balls_coords = [ frame_pose[0]['coords']
            #             for frame_pose in preproc_balls_coords]
            preproc_balls_coords = [ [pose['coords'] for pose in frame_poses]
                        for frame_poses in preproc_balls_coords]
        else:
            # No ball detected
            # preproc_balls_coords = [ [[0., 0.],[0., 0.]]
            #             for _ in range(len(video_ball_bboxes))]
            preproc_balls_coords = [ [[[0., 0.],[0., 0.]]]
                        for _ in range(len(video_balls_bboxes))]
    else:
        prefix, clip = os.path.split(video_gt.path)
        prefix, video = os.path.split(prefix)
        data_path = video+'/'+clip
        with h5py.File(hdf_filepath, "r") as f:
            preproc_balls_coords = f[data_path][()]
        preproc_balls_coords = np.asarray(preproc_balls_coords, dtype='float32')
    
    if use_center and len(preproc_balls_coords.shape) == 4:
        centers = np.transpose(preproc_balls_coords, (0,1,3,2)).mean(axis=-1)
        preproc_balls_coords = centers
    
    if interpolate: # assumes is always center
        if len(preproc_balls_coords.shape) == 4:
            raise NotImplementedError("Interpolate with non-center coords")
            
        preproc_balls_coords = interpolate_coords(preproc_balls_coords)
    
    if noise_std > 0 or noise_dr > 0:
        preproc_balls_coords = add_noise(preproc_balls_coords, noise_std, 
                                         noise_dr, video_gt.resolution)
    
    if normalization is not None:
        if use_center:
            preproc_balls_coords = preproc_balls_coords[:,:,np.newaxis,:]
        # balls_poses = video_coords_to_poses(preproc_balls_coords)
        # balls_poses = apply_norm(balls_poses, normalization, 
        #                                   video_gt)
        # preproc_balls_coords = video_poses_to_coords(balls_poses)
        
        preproc_balls_coords = apply_norm(preproc_balls_coords, normalization, 
                                          video_gt)
        if use_center:
            preproc_balls_coords = np.squeeze(preproc_balls_coords, axis=2)
    
    return preproc_balls_coords

def get_data(gt_split, pose_style, timesteps=16, skip_timesteps=None,
        add_joint_idx=True, add_body_part=True, normalization=None, 
        selected_joints=None, num_classes_indiv=None, num_classes_grp=None, 
        sample_method='central', seq_step=None, flat_seqs=False,
        group_indivs=False, num_indivs=None, add_hoop=False, add_ball=False, 
        hoops_hdf_filepath=None, balls_hdf_filepath=None, indiv_ova_class=None,
        sort_players=False, noise_std=0, noise_dr=0,
        **poses_preproc_kwargs):
    if pose_style == 'OpenPose' or pose_style == 'OpenPose-preproc':
        joint_indexing = POSE_BODY_25_BODY_PARTS
        body_parts_mapping = POSE_BODY_25_BODY_PARTS_COARSE
    elif pose_style == 'AlphaPose' or pose_style == 'AlphaPose-preproc':
        joint_indexing = ALPHA_COCO_17_BODY_PARTS
        body_parts_mapping = ALPHA_COCO_17_BODY_PARTS_COARSE
    
    if num_classes_indiv is None and num_classes_grp is None:
        raise ValueError("num_classes_indiv and num_classes_grp are both None")
    
    ### Reading video poses and objs coordinates
    all_video_poses = []
    all_video_hoops = []
    all_video_balls = []
    for video_id, video_gt in gt_split.iterrows():
        ts = timesteps
        if skip_timesteps is not None:
            ts *= skip_timesteps
        video_poses = read_video_poses(video_gt, pose_style, 
            normalization=normalization, timesteps=ts, 
            **poses_preproc_kwargs)
        # if video_poses == []:
        if np.asarray(video_poses).size == 0:
        # if not any(video_poses):
            # video_poses = [[ {'coords': np.zeros((len(joint_indexing),2))} ]]
            video_poses = np.zeros( (1,1,len(joint_indexing),2) )
        if selected_joints is not None:
            # for frame_pose in video_poses:
            #     for person in frame_pose:
            #         filter_joints(person, selected_joints, joint_indexing)
            joints_mask = [ joint_indexing.index(joint) 
                           for joint in selected_joints]
            video_poses = video_poses[:,:,joints_mask,:]
        all_video_poses.append(video_poses)
        
        if add_hoop:
            hoop_coords = get_hoop_coords(video_gt, hoops_hdf_filepath, True)
            num_frames = len(video_poses)
            if num_frames < len(hoop_coords):
                center = len(hoop_coords)//2
                central_slice = slice(center - num_frames//2,
                                      center + num_frames//2 + num_frames%2)
                hoop_coords = hoop_coords[central_slice]
            all_video_hoops.append( hoop_coords )
        
        if add_ball:
            ball_coords = get_balls_coords(video_gt, balls_hdf_filepath, 
                               use_center=True, normalization=normalization,
                               noise_std=noise_std, noise_dr=noise_dr)
            num_frames = len(video_poses)
            if num_frames < len(ball_coords):
                center = len(ball_coords)//2
                central_slice = slice(center - num_frames//2,
                                      center + num_frames//2 + num_frames%2)
                ball_coords = ball_coords[central_slice]
            all_video_balls.append( ball_coords )
    
    # Scale used for the joint and body parts indexes only
    scale = (1 if normalization != 'NTU' else 10) # unscaled or div_10
    # scale = (1 if normalization is None else 10) # unscaled or div_10
    # scale = .1 # x10: keep similar order of magnitude to the coordinates
    
    persons_info = get_persons_info(gt_split.iloc[0], prune_missing=False)
    if num_indivs is None:
        num_persons = len(persons_info)
    else:
        num_persons = num_indivs
    
    # num_objs = num_persons + add_hoop + add_ball
    num_objs = num_persons + (add_hoop or add_ball) # Objs will be merged
    # num_coords_obj = 3 # Three centers per obj, since top 3 detects
    num_coords_obj = 1 # Just one, since it is the gt annotation
    
    ### Processing data into network input
    X = []
    Y_indiv, Y_grp = [], []
    # num_joints = len(all_video_poses[0][0][0]['coords'])
    # num_dim = len(all_video_poses[0][0][0]['coords'][0])
    num_joints = all_video_poses[0].shape[2]
    num_dim = all_video_poses[0].shape[3]
    
    for i, video_poses in enumerate(all_video_poses):
        video_gt = gt_split.iloc[i]
        persons_info = get_persons_info(video_gt, prune_missing=False)
        indiv_actions = persons_info.action_id.values[:num_persons]
        Y_grp.append(video_gt.grp_activity_id)
        
        if skip_timesteps is not None:
            video_poses = video_poses[::skip_timesteps]
        
        num_frames = len(video_poses)
        all_joint_coords = video_poses[:,:num_persons]
        # all_joint_coords = np.asarray([ 
        #     [person_pose['coords'] for person_pose in frame_pose[:num_persons]] 
        #     for frame_pose in video_poses])
        # Shape: (num_frames, num_persons, num_joints, num_dim)
        if all_joint_coords.shape[1] < num_persons: 
            # Need to append person so all videos have the same number
            pad_val = num_persons - all_joint_coords.shape[1]
            pad_width = ((0,0), (0,pad_val), (0,0), (0,0))
            all_joint_coords = np.pad(all_joint_coords, pad_width=pad_width)
        
        if sort_players:
            sorted_idx = argsort_players(video_gt)
            sorted_idx = [ (idx if idx is not None else 11) 
                          for idx in sorted_idx ]
            all_joint_coords = all_joint_coords[:,sorted_idx]
            indiv_actions = indiv_actions[sorted_idx]
        
        ### Concatenating objs and appending to all_joint_coords
        if add_hoop and add_ball:
            hoop_coords = all_video_hoops[i]
            ball_coords = all_video_balls[i]
            
            pad_width = ((0,0), (0, num_coords_obj-len(hoop_coords[0])), (0,0))
            hoop_coords = np.pad(hoop_coords, pad_width=pad_width)
            
            pad_width = ((0,0), (0, num_coords_obj-len(ball_coords[0])), (0,0))
            ball_coords = np.pad(ball_coords, pad_width=pad_width)
            
            objs_coords = np.concatenate((hoop_coords, ball_coords), 1)
            all_joint_coords = append_obj(all_joint_coords, objs_coords,
                                      skip_timesteps, num_joints, num_frames)
        elif add_hoop:
            hoop_coords = all_video_hoops[i]
            all_joint_coords = append_obj(all_joint_coords, hoop_coords,
                                      skip_timesteps, num_joints, num_frames)
        elif add_ball:
            ball_coords = all_video_balls[i]
            all_joint_coords = append_obj(all_joint_coords, ball_coords,
                                      skip_timesteps, num_joints, num_frames)
        
        ### 1) Keeping only the central timesteps
        if sample_method == 'central':
            if num_frames < timesteps: # Need to pad sequence
                missing_half = (timesteps - num_frames)/2
                pad_val = int( np.ceil((timesteps - num_frames)/2) )
                pad_w_frames = (int(missing_half),int(np.ceil(missing_half)))
                pad_width = (pad_w_frames, (0,0), (0,0), (0,0))
                all_joint_coords = np.pad(all_joint_coords,pad_width=pad_width)
            if timesteps < num_frames:
                center = num_frames//2
                central_window = slice(center - timesteps//2, 
                                       center + timesteps//2 + timesteps%2)
                all_joint_coords = all_joint_coords[central_window]
                
            # Shape: (timesteps, num_objs, num_joints, num_dim)
            all_joint_coords = all_joint_coords.transpose((1,2,0,3))
            # Shape: (num_objs, num_joints, timesteps, num_dim)
            all_joint_coords = all_joint_coords.reshape(
                (num_objs, num_joints, timesteps*num_dim))
            all_joint_coords = all_joint_coords.tolist()
            
            for person_joints in all_joint_coords:
                if add_joint_idx:
                    insert_joint_idx(person_joints, num_joints, scale)
                if add_body_part:
                    insert_body_part(person_joints, num_joints, scale, 
                                     body_parts_mapping)
            if group_indivs:
                X.append([ j_coords for p_coords in all_joint_coords
                          for j_coords in p_coords])
            else:
                # Ball will be appended to each player
                if add_ball or add_hoop:
                    persons_coords = all_joint_coords[:num_persons]
                    objs_coords = all_joint_coords[num_persons:]
                    for obj_coords in objs_coords:
                        for obj_coord in obj_coords[:num_coords_obj]:
                            for p_idx in range(num_persons):
                                persons_coords[p_idx].append(obj_coord)
                    all_joint_coords = persons_coords
                
                X += all_joint_coords
            
            Y_indiv += indiv_actions.tolist()
        ### 2) Breaking the video into multiple inputs of length 'timesteps'
        elif sample_method == 'all':
            if num_frames < timesteps: # Need to pad sequence
                missing_half = (timesteps - num_frames)/2
                pad_val = int( np.ceil((timesteps - num_frames)/2) )
                pad_w_frames = (int(missing_half),int(np.ceil(missing_half)))
                pad_width = (pad_w_frames, (0,0), (0,0), (0,0))
                all_joint_coords = np.pad(all_joint_coords,pad_width=pad_width)
                num_frames = len(all_joint_coords)
            
            ## No overlap, seq_step = timesteps
            # trunc_frame = (p1_and_p2.shape[1]//timesteps)*timesteps
            # all_joint_coords = all_joint_coords[:trunc_frame].reshape(
                # (num_joints*2, trunc_frame//timesteps, timesteps*num_dim))
            # all_joint_coords = all_joint_coords.transpose((1,0,2))
            
            range_end = (num_frames - timesteps + 1)
            if seq_step is None:
                seq_step = timesteps//2
            all_seqs = [ all_joint_coords[i:i+timesteps]
                        for i in range(0, range_end, seq_step)]
            
            all_seqs = [ seq.transpose((1,2,0,3)).reshape(
                        (num_objs,num_joints,timesteps*num_dim)).tolist() 
                        for seq in all_seqs ]
            
            for seq in all_seqs:
                for person_joints in seq:
                    if add_joint_idx:
                        insert_joint_idx(person_joints, num_joints, scale)
                    if add_body_part:
                        insert_body_part(person_joints, num_joints, scale, 
                                         body_parts_mapping)
            
            ### TODO sample_method = 'all' and group_indivs = True
            if group_indivs:
                # X.append([ j_coords for p_coords in all_joint_coords
                #           for j_coords in p_coords])
                all_seqs = [ [ j_coords for p_coords in seq
                          for j_coords in p_coords] for seq in all_seqs ]
                
            if flat_seqs:
                X += all_seqs
                Y_indiv += [indiv_actions.tolist()] * len(all_seqs)
            else:
                X.append(all_seqs)
                Y_indiv.append(indiv_actions.tolist())
    
    ### Replacing dummy persons action by most common
    Y_indiv = np.asarray(Y_indiv)
    most_common = np.bincount(Y_indiv[Y_indiv!=-1]).argmax()
    Y_indiv[Y_indiv==-1] = most_common
    
    # if num_classes_indiv is None:
    #     num_classes_indiv = np.max(Y_indiv)+1
    # if num_classes_grp is None:
    #     num_classes_grp = gt_split.grp_activity_id.max()+1
    # Y_indiv = to_categorical(Y_indiv, num_classes_indiv)
    # Y_grp = to_categorical(Y_grp, num_classes_grp)
    # Y = [Y_indiv, Y_grp]
    
    Y = []
    if num_classes_grp is not None:
        Y_grp = to_categorical(Y_grp, num_classes_grp)
        Y.append(Y_grp)
    if num_classes_indiv is not None:
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
    
    # Input for the network must be (n_joints, n_samples, timesteps*num_dim)
    # if sample_method == 'central' or flat_seqs:
    if sample_method == 'central':
        X = np.asarray(X).transpose((1,0,2)).tolist()
        # Prunning padded obj coords if necessary
        # Objs coords were padded to have the same number as num_joints
        if (add_ball or add_hoop) and group_indivs:
            persons_joints = X[:num_persons*num_joints]
            objs_coords = X[num_persons*num_joints:]
            prunned_objs_coords = []
            for obj_offset in range(0, len(objs_coords), num_joints):
                prunned_objs_coords += objs_coords[obj_offset:obj_offset+num_coords_obj]
            X = persons_joints + prunned_objs_coords
    
    return X, Y
