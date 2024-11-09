import pandas as pd
import numpy as np
import os

DATA_DIR = '/usr/local/inter-rel-net-hockey/data02/volleyball/'

""" Folder structure
    [descs.csv]
    [skl.npy]
    skeletons/
        0/
            3596/
                3576_keypoints.json
                ...
            ...
        ...
        54/
    videos/
        0/
            annotations.txt
            3596/
                3576.jpg
                3577.jpg
                ...
                3596.jpg
                ...
                3616.jpg
            3646/
            3656/
            ...
        1
        2
        ...
    volleyball_tracking_annotation/
        0/ (same as videos)
"""

GRP_ACTIVITIES = ['Right set','Right spike','Right pass','Right winpoint',
                  'Left winpoint','Left pass','Left spike','Left set']

# How it is written at 'annotations.txt'
# GRP_ACTIVITIES_ANNOTATIONS = ['r_set','r_spike','r-pass','r_winpoint',
#                               'l_winpoint','l-pass','l-spike','l_set']
# Replaced '-' with '_' to make it standard
GRP_ACTIVITIES_ANNOTATIONS = ['r_set','r_spike','r_pass','r_winpoint',
                              'l_winpoint','l_pass','l_spike','l_set']
# GRP_ACTIVITIES_ANNOTATIONS = ['r_set','r_spike','r_pass','r_winpoint',
#                               'l_set','l_spike','l_pass','l_winpoint']

# Lowercase only at 'annotations.txt'
IND_ACTIONS = ['Waiting','Setting','Digging','Falling','Spiking','Blocking',
               'Jumping','Moving','Standing']
# Lowercase only at 'annotations.txt'
IND_ACTIONS_ANNOTATIONS = ['waiting','setting','digging','falling','spiking',
                           'blocking','jumping','moving','standing']

ALL_VIDEOS = list(range(55))
TRAIN_VIDEOS = [1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,
                52,53,54]
VALIDATION_VIDEOS = [0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
TEST_VIDEOS = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]

HD_VIDEOS = [2, 37, 38, 39, 40, 41, 44, 45] # 1920x1080, rest is 1280x720

def parse_line_annotations(line_str):
    """
    Parse string line of file 'annotations.txt'

    Parameters
    ----------
    line_str : str
        String text to be parsed.

    Returns
    -------
    line_annotations : list
        List of annotations with: frame_num, grp_activity and players info.

    """
    frame, grp_act, *players = line_str.split()
    frame = int(frame.split('.')[0])
    
    grp_act = grp_act.replace('-', '_')
    
    players = np.array(players).reshape((-1,5)) # ?? players, 5 info fields
    players_info = []
    for idx, player_info in enumerate(players):
        x_pos, y_pos, width, height, p_action = player_info
        width = int(width)
        height = int(height)
        x_pos = int(x_pos)
        y_pos = int(y_pos)
        # players_info.append([x_pos, y_pos, width, height, p_action])
        players_info += [x_pos, y_pos, width, height, p_action]
    for missing_idx in range(idx+1,12):
        players_info += [0, 0, 0, 0, 'missing']
    
    return [frame, grp_act] + players_info

def parse_annotations(annotations_filepath):
    """
    Parse 'annotations.txt'.

    Parameters
    ----------
    annotations_filepath : string
        Path to annotations file.

    Returns
    -------
    annotations : matrix
        Annotations for all frames in the file.

    """
    with open(annotations_filepath) as annotations_file:
        annotations = []
        line_str = annotations_file.readline()
        # i = 0
        while line_str != '':
            # print(i)
            # i += 1
            annotations.append(parse_line_annotations(line_str))
            line_str = annotations_file.readline()
    return annotations

def get_players_info(video_gt, prune_missing=True):
    players_info = []
    for p_id in range(12):
        p_label = 'p{}'.format(p_id+1)
        p_cols = [ col for col in video_gt.index 
                  if col.startswith(p_label+'_')]
        p_info = list(video_gt[p_cols].values)
        players_info.append(p_info)
    players_info_fields = ['x','y','width','height','action','action_id']
    players_info = pd.DataFrame(players_info, columns=players_info_fields)
    if prune_missing:
        players_info = players_info[players_info.action != 'missing']
    return players_info

def get_ground_truth(data_dir=DATA_DIR, reload=False, insert_event_bboxes=False):
    """
    Read ground_truth file if it exists, create it otherwise.

    Parameters
    ----------
    data_dir : string, optional
        Path to the data dir. The default is DATA_DIR hard-coded value.

    Returns
    -------
    ground_truth : pandas.DataFrame
        Dataset ground_truth.

    """
    if os.path.exists(data_dir+'/descs.csv') and not reload:
        ground_truth = pd.read_csv(data_dir+'/descs.csv', index_col=0)
    else:
        all_annotations = []
        for video_id in ALL_VIDEOS:
            video_annotations_filepath = os.path.join(data_dir,'videos',
                                          str(video_id), 'annotations.txt')
            annotations = parse_annotations(video_annotations_filepath)
            
            path_tpl = os.path.join(data_dir,'skeletons', str(video_id), '{}')
            # all_annotations += [ [video_id] + a for a in annotations]
            all_annotations += [ [video_id] + a + [path_tpl.format(a[0])] 
                                for a in annotations]
            # TODO add event_bboxes here?
        
        players_info_fields = ['x','y','width','height','action']
        df_columns = ['video','frame','grp_activity']
        for player_id in range(12):
            for f in players_info_fields:
                df_columns.append(('p{}_'+f).format(player_id+1))
        df_columns.append('path')
        ground_truth = pd.DataFrame(all_annotations, columns=df_columns)
        ground_truth.sort_values(['video','frame'], inplace=True)
        ground_truth.reset_index(drop=True, inplace=True)
        ground_truth.to_csv(data_dir+'/descs.csv')
    
    ground_truth['grp_activity_id'] = [GRP_ACTIVITIES_ANNOTATIONS.index(label)
                               for label in ground_truth.grp_activity.values ]
    
    sub_acts = ground_truth.grp_activity.str.split('_', expand=True)
    sub_acts[0] = sub_acts[0].str.replace('r', 'right')
    sub_acts[0] = sub_acts[0].str.replace('l', 'left')
    ground_truth['grp_sub_activity_1'] = sub_acts[1]
    ground_truth['grp_sub_activity_2'] = sub_acts[0]
    for sub_act in ['grp_sub_activity_1', 'grp_sub_activity_2']:
        sorted_labels = sorted(ground_truth[sub_act].unique())
        ground_truth[sub_act+'_id'] = [ sorted_labels.index(label)
                                       for label in ground_truth[sub_act]]
    
    for player_id in range(12):
        p_action_field = 'p{}_action'.format(player_id+1)
        ground_truth[p_action_field+'_id'] = [
            (IND_ACTIONS_ANNOTATIONS.index(label) if label !='missing' else -1)
            for label in ground_truth[p_action_field].values ]
    
    ground_truth['path'] = [ os.path.join(data_dir, 'skeletons', 
                              str(video_gt.video), str(video_gt.frame))
                            for _, video_gt in ground_truth.iterrows() ]
    ground_truth['video_path'] = [ os.path.join(data_dir, 'videos', 
                              str(video_gt.video), str(video_gt.frame))
                            for _, video_gt in ground_truth.iterrows() ]
    
    ground_truth['split'] = 'validation'
    ground_truth.loc[ground_truth.video.isin(TRAIN_VIDEOS), 'split'] = 'train'
    ground_truth.loc[ground_truth.video.isin(TEST_VIDEOS), 'split'] = 'test'
    
    # ground_truth['resolution'] = '1280x720' # 'sd' # 1280x720
    # ground_truth.loc[ground_truth.video.isin(HD_VIDEOS), 'resolution'] = '1920x1080'
    ground_truth['resolution'] = 'sd' # 1280x720
    ground_truth.loc[ground_truth.video.isin(HD_VIDEOS), 'resolution'] = 'hd'
    
    # if insert_event_bboxes:
    #     ground_truth = add_mult_event_bboxes(ground_truth)
    
    return ground_truth

def get_split_gt(split, dataset_fold=None):
    if split == 'train':
        split_gt = get_train_gt(dataset_fold)
    elif split == 'validation':
        split_gt = get_val_gt(dataset_fold)
    elif split == 'test':
        split_gt = get_test_gt(dataset_fold)
    return split_gt

def get_train_gt(dataset_fold=None):
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.video.isin(TRAIN_VIDEOS)]
    return gt_split

def get_val_gt(dataset_fold=None):
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.video.isin(VALIDATION_VIDEOS)]
    return gt_split

def get_test_gt(dataset_fold=None):
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.video.isin(TEST_VIDEOS)]
    return gt_split

def get_event_bboxes(video_gt):
    """
    - Each annotation line consists of the following components: 
        {player ID} {xmin} {ymin} {xmax} {ymax} {frame ID} {lost} {grouping} 
        {generated} {individual action label}
    - xmin, ymin, xmax, ymax: The bounding box of this player.
    - lost: If 1, the annotated bounding box is outside of the field of view.
    - grouping: If 1, this player is involved in a primary group activity.
    - generated: If 1, the bounding box was automatically interpolated.
    - individual action label: The individual action label of each player, 
    which was given in the original annotation of the Volley ball dataset.

    Parameters
    ----------
    video_gt : pandas Series
        DESCRIPTION.

    Returns
    -------
    event_gt_bboxes : dict of 'frame_num': bboxes
        bboxes for all players in multiple frames around the center.

    """
    trck_filepath = os.path.join(DATA_DIR, 'volleyball_tracking_annotation', 
         str(video_gt.video), str(video_gt.frame), str(video_gt.frame)+'.txt')
    
    names = ['player_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 
              'grouping', 'generated', 'action']
    trck_gt = pd.read_csv(trck_filepath, index_col=False, sep=' ', names=names,
                     dtype={'lost': bool, 'grouping': bool, 'generated': bool})
    
    event_gt_bboxes = {}
    for frame, frame_gt in trck_gt.groupby('frame'):
        frame_bboxes = frame_gt[['xmin','ymin','xmax','ymax']].values
        frame_bboxes[:,2] = frame_bboxes[:,2] - frame_bboxes[:,0]
        frame_bboxes[:,3] = frame_bboxes[:,3] - frame_bboxes[:,1]
        
        # event_gt_bboxes[frame] = frame_bboxes.tolist()
        # event_gt_bboxes[frame-video_gt.frame+20] = frame_bboxes.tolist()
        event_gt_bboxes[frame-video_gt.frame+21] = frame_bboxes.tolist()
    
    # event_gt_bboxes = [[] for _ in range(11)] + list(event_gt_bboxes.values())
    # event_gt_bboxes += [[] for _ in range(10)] 
    
    return event_gt_bboxes

def add_event_bboxes(video_gt):
    event_bboxes = get_event_bboxes(video_gt)
    
    with pd.option_context('mode.chained_assignment', None):
        video_gt['event_bboxes'] = event_bboxes
    
    return video_gt

def add_mult_event_bboxes(gt_split):
    split_event_bboxes = [get_event_bboxes(video_gt)
                          for _, video_gt in gt_split.iterrows()]
    
    with pd.option_context('mode.chained_assignment', None):
        gt_split['event_bboxes'] = split_event_bboxes
    
    return gt_split

def add_event_rois(video_gt):
    from misc.utils import merge_bboxes
    
    if 'event_bboxes' in video_gt:
        event_bboxes = video_gt['event_bboxes']
    else:
        event_bboxes = get_event_bboxes(video_gt)
    
    big_bboxes = [ merge_bboxes(frm_ply_bboxes, ignore_negs_bbox=False) for
                  frm_ply_bboxes in event_bboxes.values()]
    
    num_frames = 41
    offset_before = list(event_bboxes.keys())[0]
    last_frame_idx = list(event_bboxes.keys())[-1]
    offset_after = num_frames -last_frame_idx -1
    
    event_rois = [[]]*offset_before + big_bboxes + [[]]*offset_after
    
    with pd.option_context('mode.chained_assignment', None):
        video_gt['event_rois'] = event_rois
    
    return video_gt
















