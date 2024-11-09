# %% Imports

from time import sleep
from misc import data_io, utils
from datasets import Volleyball, ncaa, cad, cad_new
import numpy as np
import h5py, os

import progressbar

# %% Parallel Detections - multiprocessing 
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue

from multiprocessing import Process, Queue

## get_hoop_coords
# def do_work(video_gt):
#     video_gt['event_rois'] = ncaa.get_event_rois(video_gt)
#     # I deleted something important here! missing data_io.getxxx
#     hoop_coords =  np.asarray(hoop_coords).astype('float32')
#     return hoop_coords

## estimate_motion
# def do_work(video_gt):
#     video_frames = utils.read_video(video_gt.video_path)
#     cam_motions = utils.estimate_motion(video_frames)
#     cam_motions =  np.asarray(cam_motions).astype('float32')
#     return cam_motions

## estimate_motion with skeleton
# def do_work(video_gt):
#     video_poses = data_io.read_video_poses(video_gt, pose_style='OpenPose-preproc')
#     norm_data = data_io.apply_center_normalization(video_poses, 
#                                                    return_estimations=True)
#     norm_video_poses, cumsum_cam_motions, new_center = norm_data
#     # cam_motions = cumsum_cam_motions[:1] + np.diff(cumsum_cam_motions, 
#     #                                                axis=0).tolist()
#     # data = cam_motions
#     data = new_center
    
#     data =  np.asarray(data).astype('float32')
#     return data

## Normalized skeletons
# def do_work(video_gt):
#     video_poses = data_io.read_video_poses(video_gt, pose_style='OpenPose-preproc')
#     norm_data = data_io.apply_center_normalization(video_poses, 
#                                                    return_estimations=True)
#     norm_video_poses, cumsum_cam_motions, new_center = norm_data
#     # cam_motions = cumsum_cam_motions[:1] + np.diff(cumsum_cam_motions, 
#     #                                                axis=0).tolist()
#     # data = cam_motions
#     data = new_center
    
#     data =  np.asarray(data).astype('float32')
#     return data

## Pre-proc skeletons
def do_work(video_gt):
    # v1 - Not sure
    # preproc_kwargs = dict()
    preproc_kwargs = dict(timesteps=40)
    
    # v2
    # preproc_kwargs = dict(
    #     per_player = True, # [False]
    #     track = True, 
    #     track_metric = 'area_intersection', 
    #     extrapolate = False, # [True]
    #     interpolate = True, # [True]
        
    #     prune = True, # [True]
    # )
    # video_gt = Volleyball.add_event_bboxes(video_gt)
    
    # v3
    # preproc_kwargs = dict(
    #     filter_rois = True, # [False]
    #     track = True, 
    #     track_metric = 'area_intersection', 
    #     extrapolate = False, # [True]
    #     interpolate = True, # [True]
        
    #     prune = True, # [True]
    #     prune_metric = 'match_event_bboxes', 
    # )
    # video_gt = Volleyball.add_event_bboxes(video_gt)
    # video_gt = Volleyball.add_event_rois(video_gt)
    
    # 'OpenPose' | AlphaPose
    video_coords = data_io.read_video_poses(video_gt, pose_style='OpenPose',
                                            **preproc_kwargs)
    
    video_coords =  np.asarray(video_coords).astype('float32')
    return video_coords

def worker(input_q, output_q):
    while not input_q.empty():
        video_gt = input_q.get(timeout=1)
        
        coords = do_work(video_gt)
        
        prefix, clip = os.path.split(video_gt.path)
        prefix, video = os.path.split(prefix)
        data_path = '{}/{}'.format(video, clip)
        
        output = {data_path: coords}
        output_q.put(output)

if __name__ == '__main__': # This is required to avoid workers getting in
# if False:
    # print("> NCAA Basketball - Hoops Multiprocessing")
    # print("> NCAA Basketball - Estimate Motions Multiprocessing")
    # print("> Volleyball - Estimate Motions Multiprocessing")
    # print("> Volleyball - Estimate Center Multiprocessing")
    # print("> CAD - Pre-proc skeletons Multiprocessing")
    print("> CAD New - Pre-proc skeletons Multiprocessing")
    dataset = cad_new # Volleyball | cad
    
    input_q = Queue()
    output_q = Queue()

    gt = dataset.get_ground_truth()
    # gt = gt.head(10)
    
    ### file path definition
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'hoops_v3.1_home.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'cam_motions-skels.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'center-skels.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'cam_motions_v1_home.hdf5')
    hdf_filepath = os.path.join(dataset.DATA_DIR, 'skeletons-ts_40_new.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'alphapose-v3.hdf5')
    print("hdf_filepath:", hdf_filepath)
    
    stop = False
    if os.path.exists(hdf_filepath):
        print("hdf file already exists.")
        answer = input("OK to continue [Y/N]? ").lower()
        if answer != 'y':
            print("Will quit without running program...")
            stop = True
    
    if not stop:
        with h5py.File(hdf_filepath, "a") as f:
            for idx, video_gt in gt.iterrows():
                prefix, clip = os.path.split(video_gt.path)
                prefix, video = os.path.split(prefix)
                data_path = '{}/{}'.format(video, clip)
                if data_path not in f:
                    # print(video_gt.name,video_gt.YoutubeId,video_gt.EventEndTime)
                    input_q.put(video_gt)
        size_queue = input_q.qsize()
        
        num_workers = 12 # 225 has 24 cores
        print("Number of workers to use:", num_workers)
        for i in range(num_workers):
            Process(target=worker, args=(input_q, output_q)).start()
        
        progbar = progressbar.ProgressBar(max_value=size_queue)
        for idx in range(size_queue):
            sleep(.05)
            output = output_q.get(timeout=60)
            progbar.update(idx)
            
            data_path, coords = list(output.items())[0]
            # print(data_path)
            with h5py.File(hdf_filepath, "r+") as f:
                f.create_dataset(data_path, data=coords)
        progbar.finish()

# %% Parallel Hoops - Threads
# https://docs.python.org/3/library/queue.html
# Slower

# if True:
if False:
    import threading, queue
    
    input_q = queue.Queue()
    output_q = queue.Queue()
        
    def worker_thread():
        # while True:
            # video_gt = input_q.get(timeout=60)
        while not input_q.empty():
            video_gt = input_q.get(timeout=1)
            
            coords = do_work(video_gt)
            
            prefix, clip = os.path.split(video_gt.path)
            prefix, video = os.path.split(prefix)
            data_path = '{}/{}'.format(video, clip)
            
            output = {data_path: coords}
            output_q.put(output)
            input_q.task_done()
        
    print("> NCAA Basketball - Hoops")
    gt = ncaa.get_ground_truth()
    # gt = gt.head(12)
    
    # hdf_filepath = os.path.join(ncaa.DATA_DIR, 'hoops_test.hdf5')
    hdf_filepath = os.path.join(ncaa.DATA_DIR, 'hoops_v3.1_home.hdf5')
    print("hdf_filepath:", hdf_filepath)
    
    with h5py.File(hdf_filepath, "a") as f:
        for idx, video_gt in gt.iterrows():
            prefix, clip = os.path.split(video_gt.path)
            prefix, video = os.path.split(prefix)
            data_path = '{}/{}'.format(video, clip)
            if data_path not in f:
                # print(video_gt.name,video_gt.YoutubeId,video_gt.EventEndTime)
                input_q.put(video_gt)
    size_queue = input_q.qsize()
    
    num_worker_threads = 6 # 225 has 24 cores
    threads = []
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker_thread)
        t.start()
        threads.append(t)
    
    progbar = progressbar.ProgressBar(max_value=size_queue)
    for idx in range(size_queue):
        output = output_q.get(timeout=60)
        progbar.update(idx)
        
        data_path, coords = list(output.items())[0]
        # print(data_path)
        with h5py.File(hdf_filepath, "a") as f:
            f.create_dataset(data_path, data=coords)
    progbar.finish()
    
    for t in threads:
        t.join()

# %% Running for NCAA Basketball - Balls

# if True:
if False:
    print("> NCAA Basketball - Balls")
    gt = ncaa.get_ground_truth()
    
    hdf_filepath = os.path.join(ncaa.DATA_DIR, 'balls_home.hdf5')
    hoop_hdf_filepath = os.path.join(ncaa.DATA_DIR, 'hoops_home.hdf5')
    print("hdf_filepath:", hdf_filepath)
    
    with h5py.File(hdf_filepath, "a") as f:
        progbar = progressbar.ProgressBar(max_value=len(gt))
        for idx, video_gt in gt.iterrows():
        # for idx, video_gt in gt[gt.index==11577].iterrows():
        # for idx, video_gt in gt.head(5).iterrows():
            progbar.update(idx)
            # print(idx)
            prefix, clip = os.path.split(video_gt.path)
            prefix, video = os.path.split(prefix)
            data_path = '{}/{}'.format(video, clip)
            if data_path not in f:
                hoops_coords = data_io.get_hoop_coords(video_gt, 
                                               hdf_filepath=hoop_hdf_filepath)
                video_gt['event_rois'] = ncaa.get_event_rois(video_gt)
                hoop_coords =  data_io.get_balls_coords(video_gt, hoops_coords)
                hoop_coords =  np.asarray(hoop_coords).astype('float32')
                f.create_dataset(data_path, data=hoop_coords)
        progbar.finish()

# %% Running for NCAA Basketball - Hoops

# if True:
if False:
    print("> NCAA Basketball - Hoops")
    gt = ncaa.get_ground_truth()
    
    hdf_filepath = os.path.join(ncaa.DATA_DIR, 'hoops_v3_home.hdf5')
    print("hdf_filepath:", hdf_filepath)
    
    with h5py.File(hdf_filepath, "a") as f:
        progbar = progressbar.ProgressBar(max_value=len(gt))
        for idx, video_gt in gt.iterrows():
        # for idx, video_gt in gt[gt.index==11577].iterrows():
        # for idx, video_gt in gt.head(5).iterrows():
            progbar.update(idx)
            # print(idx)
            prefix, clip = os.path.split(video_gt.path)
            prefix, video = os.path.split(prefix)
            data_path = '{}/{}'.format(video, clip)
            if data_path not in f:
                video_gt['event_rois'] = ncaa.get_event_rois(video_gt)
                hoop_coords =  data_io.get_hoop_coords(video_gt)
                hoop_coords =  np.asarray(hoop_coords).astype('float32')
                f.create_dataset(data_path, data=hoop_coords)
        progbar.finish()

# %% Reading hdf file

# python src/preprocess_detections.py > checking_hoops.txt 2>&1

# if True:
if False:
    print("> NCAA Basketball - Reading hdf file")
    # dataset = ncaa
    dataset = Volleyball
    gt = dataset.get_ground_truth()
    
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'hoops.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'hoops_v3.1.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'balls.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'hoops_test.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'cam_motions_v1.hdf5')
    # hdf_filepath = os.path.join(dataset.DATA_DIR, 'cam_motions.hdf5')
    hdf_filepath = os.path.join(dataset.DATA_DIR, 'cam_motions-skels.hdf5')
    print("hdf_filepath:", hdf_filepath)
    
    with h5py.File(hdf_filepath, "r") as f:
        # for idx, video_gt in gt.iterrows():
        for idx, video_gt in gt.head(15).iterrows():
            prefix, clip = os.path.split(video_gt.path)
            prefix, video = os.path.split(prefix)
            data_path = '{}/{}'.format(video, clip)
            
            print(data_path)
            if data_path in f:
                # print(f[data_path])
                # print(f[data_path][()])
                print('\t', f[data_path].shape)
