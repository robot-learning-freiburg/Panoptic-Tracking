import argparse
import glob
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
import numpy as np
import sys
import torch
import tqdm
import umsgpack
import json
from collections import defaultdict

parser = argparse.ArgumentParser(description="MOTSP Evaluation")
parser.add_argument("--gt_dir", type=str, help="Ground Truth directory")
parser.add_argument("--prediction_dir", type=str, help="Prediction directory")

def main(args):
    prediction_files = []
    gt_files = []
    seq_ids = []
    seq = {} 
    count_seq = 0
    for _file in listdir(args.gt_dir):
        if 'metadata' in _file:
            continue
        elif _file.split('_')[0] not in seq:
            seq[_file.split('_')[0]] = count_seq*1000000 
            count_seq += 1

        gt_files.append(path.join(args.gt_dir, _file))
        seq_ids.append(_file.split('_')[0])
        if path.exists(path.join(args.prediction_dir, _file)):
            prediction_files.append(path.join(args.prediction_dir, _file))   
        else:
            print ('prediction doesn\'t exist for {}'.format(path.join(args.prediction_dir, _file)))     
    
    with open(path.join(args.gt_dir, "metadata.bin"), "rb") as fid:
        meta = umsgpack.unpack(fid, encoding="utf-8")

    num_stuff = meta['num_stuff']
    num_classes =  meta['num_stuff'] + meta['num_thing']
    panoptic_buffer = torch.zeros(4, num_classes, dtype=torch.double)
    seq_trajectories = defaultdict(list)
    class_trajectories = defaultdict(list)
    iou_trajectories = defaultdict(list)

    #Accumulation
    count = 0 
    for gt, prediction, seq_id in zip(gt_files, prediction_files, seq_ids):
        gt_i = np.load(gt)
        prediction_i = np.load(prediction)
        msk_gt, cat_gt, track_gt = get_processing_format(gt_i, num_stuff, seq[seq_id])        
        msk_pred, cat_pred, track_pred = get_processing_format(prediction_i, num_stuff, seq[seq_id])    
        iou, tp, fp, fn, seq_trajectories, class_trajectories, iou_trajectories = panoptic_compute(msk_gt, cat_gt, track_gt, 
                                                                                  seq_trajectories, class_trajectories, 
                                                                                  iou_trajectories, msk_pred, cat_pred, track_pred, 
                                                                                  num_classes, num_stuff)
        panoptic_buffer += torch.stack((iou, tp, fp, fn), dim=0)
        count += 1
        print("\rFiles Processed: {}".format(count), end=' ')
        sys.stdout.flush()
    print ()
    MOTSA, sMOTSA, MOTSP, PTQ, sPTQ = get_MOTSP_metrics(panoptic_buffer, num_stuff, seq_trajectories, class_trajectories, iou_trajectories)
    print_results({'MOTSA':MOTSA, 'sMOTSA':sMOTSA, 'MOTSP':MOTSP, 'PTQ':PTQ, 'sPTQ':sPTQ}, meta)
    


def print_results(metrics, meta):
    json_save = {}   
    num_stuff = meta['num_stuff']
    col = '|'
    space = ' '
    line = (space*2) + 'metric' + (space*2) + col + (space*2) + 'stuff' + (space*2) + col + (space*2) + 'thing' + (space*2) + col + (space*2) + 'all' + (space*2) + col
    print ()
    print (line)
    print ('-'*39)
    for metric in metrics:
        stuff = '----'
        thing = str(round(metrics[metric].mean().item(), 2)*100)
        all_ = thing
        if metric in ['PTQ', 'sPTQ']:
            stuff = str(round(metrics[metric][:num_stuff].mean().item(), 2)*100)
            thing = str(round(metrics[metric][num_stuff:].mean().item(), 2)*100)
            all_ = str(round(metrics[metric].mean().item(), 2)*100)
        json_save[metric] = [stuff, thing, all_]
        line = (space*2) + metric + (space*(8-len(metric))) + col + (space*2) + stuff + (space*(7-len(stuff))) + col + (space*2) + thing + (space*(7-len(thing))) + col + (space*2) + all_ + (space*(5-len(all_))) + col
        print (line)
 
    with open('results.json','w') as write_json:
        json.dump(json_save , write_json) 

def get_MOTSP_metrics(panoptic_buffer, num_stuff, seq_trajectories, class_trajectories, iou_trajectories):
    size = panoptic_buffer[0].shape[0]
    IDS, softIDS = compute_ids(seq_trajectories, class_trajectories, iou_trajectories, panoptic_buffer[0].shape[0])
    
    MOTSA = (torch.max(panoptic_buffer[1][num_stuff:]-panoptic_buffer[2][num_stuff:]-IDS[num_stuff:],torch.zeros((size-num_stuff,), dtype=torch.double)))/(panoptic_buffer[1][num_stuff:]+panoptic_buffer[3][num_stuff:]+1e-8)
    
    sMOTSA =(torch.max(panoptic_buffer[0][num_stuff:]-panoptic_buffer[2][num_stuff:]-IDS[num_stuff:],torch.zeros((size-num_stuff,),dtype=torch.double)))/(panoptic_buffer[1][num_stuff:]+panoptic_buffer[3][num_stuff:]+1e-8)

    MOTSP = (panoptic_buffer[0][num_stuff:])/(panoptic_buffer[1][num_stuff:]+1e-8) 

 
    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.

    PTQ = (panoptic_buffer[0]-IDS) / denom 
    sPTQ = (panoptic_buffer[0]-softIDS) / denom  
   
    
    return MOTSA, sMOTSA, MOTSP, PTQ, sPTQ 


def compute_ids(seq_trajectories, class_trajectories, iou_trajectories, size):
    id_switches =torch.zeros((size,), dtype=torch.double)
    soft_id_switches =torch.zeros((size,), dtype=torch.double)
    id_fragments = 0 #no use for now
 
    if len(seq_trajectories) != 0:
      for g,cl,iou in zip(seq_trajectories.values(),class_trajectories.values(), iou_trajectories.values()):
      # all frames of this gt trajectory are not assigned to any detections
       if all([this == -1 for this in g]):
        continue
      # compute tracked frames in trajectory
       last_id = g[0]
      # first detection (necessary to be in gt_trajectories) is always tracked
       tracked = 1 if g[0] >= 0 else 0
       for f in range(1, len(g)):
        if last_id != g[f] and last_id != -1 and g[f] != -1:
          id_switches[cl[f]] += 1
          soft_id_switches[cl[f]] += iou[f]
        if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
          id_fragments += 1
        if g[f] != -1:
          tracked += 1
          last_id = g[f]
      # handle last frame; tracked state is handled in for loop (g[f]!=-1)
       if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
        id_fragments += 1
    else:
       print ('something is wrong')
    return id_switches, soft_id_switches


def panoptic_compute(msk_gt, cat_gt, track_gt, seq_trajectories, class_trajectories, iou_trajectories, msk_pred, cat_pred, track_pred, num_classes, _num_stuff):
    cat_gt = torch.from_numpy(cat_gt).long()   
    msk_gt = torch.from_numpy(msk_gt).long()  
    track_gt = torch.from_numpy(track_gt).long()     
    
    for cat_id, track_id in zip(cat_gt,track_gt):
        if track_id != 0 :
            seq_trajectories[int(track_id.numpy())].append(-1)
            iou_trajectories[int(track_id.numpy())].append(-1)
            class_trajectories[int(track_id.numpy())].append(cat_id)
    
    msk_pred = torch.from_numpy(msk_pred).long()
    cat_pred = torch.from_numpy(cat_pred).long() 
    track_pred = torch.from_numpy(track_pred).long()     

    iou = msk_pred.new_zeros(num_classes, dtype=torch.double)
    tp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fn = msk_pred.new_zeros(num_classes, dtype=torch.double)

    if cat_gt.numel()>1:
        msk_gt = msk_gt.view(-1)
        msk_pred = msk_pred.view(-1)

        confmat = msk_pred.new_zeros(cat_gt.numel(), cat_pred.numel(), dtype=torch.double)
 
        confmat.view(-1).index_add_(0, msk_gt * cat_pred.numel() + msk_pred,
                                    confmat.new_ones(msk_gt.numel()))
        num_pred_pixels = confmat.sum(0)
        valid_fp = (confmat[0] / num_pred_pixels) <= 0.5

        # compute IoU without counting void pixels (both in gt and pred)
        _iou = confmat / ((num_pred_pixels - confmat[0]).unsqueeze(0) + confmat.sum(1).unsqueeze(1) - confmat)

        # flag TP matches, i.e. same class and iou > 0.5
        matches = ((cat_gt.unsqueeze(1) == cat_pred.unsqueeze(0)) & (_iou > 0.5))
        # remove potential match of void_gt against void_pred
        matches[0, 0] = 0

        _iou = _iou[matches]
        tp_i, _ = matches.max(1)
        fn_i = ~tp_i
        fn_i[0] = 0  # remove potential fn match due to void against void
        fp_i = ~matches.max(0)[0] & valid_fp
        fp_i[0] = 0  # remove potential fp match due to void against void

        # Compute per instance classes for each tp, fp, fn
        tp_cat = cat_gt[tp_i]
        fn_cat = cat_gt[fn_i]
        fp_cat = cat_pred[fp_i]
         
        match_ind = torch.nonzero(matches)
        for r in range(match_ind.shape[0]):
            if track_gt[match_ind[r,0]]!=0 and track_gt[match_ind[r,0]]>=_num_stuff:
                 seq_trajectories[int(track_gt[match_ind[r,0]].numpy())][-1] = int(track_pred[match_ind[r,1]].cpu().numpy())    
                 iou_trajectories[int(track_gt[match_ind[r,0]].numpy())][-1] = float(_iou[r].cpu().numpy())    
        if tp_cat.numel() > 0:
            tp.index_add_(0, tp_cat, tp.new_ones(tp_cat.numel()))
        if fp_cat.numel() > 0:
            fp.index_add_(0, fp_cat, fp.new_ones(fp_cat.numel()))
        if fn_cat.numel() > 0:
            fn.index_add_(0, fn_cat, fn.new_ones(fn_cat.numel()))
        if tp_cat.numel() > 0:
            iou.index_add_(0, tp_cat, _iou)

    return iou, tp, fp, fn, seq_trajectories, class_trajectories, iou_trajectories  

def get_processing_format(x, num_stuff, offset):
    x =  x.reshape(-1, 2)
    msk = np.zeros(x.shape[0], np.int32)     
    cat = [255]
    track_id = [0]
    ids = np.unique(x[:,0])
    for id_i in ids: 
        if id_i == 255:
            continue

        elif id_i < num_stuff:
            cls_i = id_i
            iss_instance_id = len(cat)
            mask_i = x[:,0] == cls_i
            cat.append(cls_i)
            track_id.append(0)
            msk[mask_i] = iss_instance_id
        else:
            t_ids = np.unique(x[x[:,0] == id_i,1])
            for t_i in t_ids:
                cls_i = id_i
                iss_instance_id = len(cat)
                mask_i = x[:,1] == t_i
                cat.append(cls_i)
                track_id.append(t_i+offset)
                msk[mask_i] = iss_instance_id
     
    return msk, np.array(cat), np.array(track_id)


if __name__ == "__main__":
    main(parser.parse_args())
    
