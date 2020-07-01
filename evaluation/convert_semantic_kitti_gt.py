import argparse
import glob
import json
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir

from laserscan import LaserScan, SemLaserScan
import numpy as np
import tqdm
import umsgpack
import yaml

parser = argparse.ArgumentParser(description="Convert Semantic Kitti to MOTSP evaluation format")
parser.add_argument("--root_dir", type=str, help="Root directory of Semantic Kitti")
parser.add_argument("--out_dir", type=str, help="Output directory")
parser.add_argument("--config", type=str, help="Semantic Kitti labels config file")
parser.add_argument("--split", type=str, help="train or valid", default='valid')
parser.add_argument("--depth_cutoff", type=float, help="depth beyond which gt label are not considered for evaluation", default='-1')
parser.add_argument("--point_cloud", type=bool, help="pointcloud or projection for evaluation", default='false')


def main(args):
    print("Loading Kitti-lidar from", args.root_dir)

    try:
        print("Opening config file %s" % args.config)
        cfg = yaml.safe_load(open(args.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    num_stuff, num_thing = _get_meta(cfg)
    _mk_dir(args.out_dir)
    split = cfg['split'][args.split]
    _list = _get_list(args.root_dir, split, cfg)   

    worker = _Worker(args.root_dir, args.out_dir, args.depth_cutoff, args.point_cloud)
    
    with Pool(initializer=_init_counter, initargs=(_Counter(0),)) as pool:
        total = len(_list)
        for feedback in tqdm.tqdm(pool.imap(worker, _list, 10), total=total):
                nothing = feedback
    
    # Write meta-data
    print("Writing meta-data")
    meta = {
            "num_stuff": num_stuff,
            "num_thing": num_thing,
            "categories": [],
            "palette": [],
            "original_ids": []
    }

    for lbl in range(num_thing+num_stuff):
        if  lbl != 255 :
            meta["categories"].append(cfg['labels'][cfg['learning_map_inv'][lbl]])
            meta["palette"].append(cfg['color_map'][cfg['learning_map_inv'][lbl]])
            meta["original_ids"].append(cfg['learning_map_inv'][lbl])
 
    with open(path.join(args.out_dir, "metadata.bin"), "wb") as fid:
        umsgpack.dump(meta, fid, encoding="utf-8")   
       
  
class _Worker():
    def __init__(self, base_dir, out_dir, depth_cutoff, point_cloud):
        self.base_dir = base_dir
        self.out_dir = out_dir
        self.depth_cutoff = depth_cutoff
        self.depth_flag = True
        self.point_cloud = point_cloud
        if self.depth_cutoff <=0:
            self.depth_flag = False
   
    def __call__(self, label_desc):
        label_dir, label_id, cfg = label_desc        
        laser = SemLaserScan(sem_color_dict=cfg['color_map'], project=True, H=cfg['sensor']['img_prop']['height'], W=cfg['sensor']['img_prop']['width'], fov_up=cfg['sensor']['fov_up'], fov_down=cfg['sensor']['fov_down'])
        laser.open_scan(path.join(self.base_dir,label_dir,'velodyne',label_id.split('_')[1]+'.bin')) 
        laser.open_label(path.join(self.base_dir,label_dir,'labels',label_id.split('_')[1]+'.label'))
       
        if self.point_cloud:
            ids = np.unique(laser.inst_label)
            ids_sem = np.unique(laser.sem_label)
            depth = laser.unproj_range
            lbl_out = np.zeros(laser.inst_label.shape +(2,), np.int32)
        else:
            ids = np.unique(laser.proj_inst_label)
            ids_sem = np.unique(laser.proj_sem_label)
            depth = laser.proj_range
            lbl_out = np.zeros(laser.proj_inst_label.shape +(2,), np.int32)

        for id_sem in ids_sem:
            cls_i = id_sem
            iss_class_id = cfg['learning_map'][cls_i]
            if cfg['instances'][iss_class_id]:
                continue
                
            else:
                if self.point_cloud:
                    lbl_out[laser.sem_label == cls_i, 0] = iss_class_id
                else: 
                    lbl_out[laser.proj_sem_label == cls_i, 0] = iss_class_id
        
        if self.point_cloud and self.depth_flag:
            laser.inst_label[depth>self.depth_cutoff]=0 
            laser.sem_label[depth>self.depth_cutoff]=0       
            ids = np.unique(laser.inst_label)
            ids_sem = np.unique(laser.sem_label)   
        
        elif self.depth_flag:  
            laser.proj_inst_label[depth>self.depth_cutoff]=0 
            laser.proj_sem_label[depth>self.depth_cutoff]=0       
            ids = np.unique(laser.proj_inst_label)
            ids_sem = np.unique(laser.proj_sem_label)   

        for id_sem in ids_sem:
            cls_i = id_sem
            iss_class_id = cfg['learning_map'][cls_i]
            
            if cfg['instances'][iss_class_id]:
                for id_ in ids:
                     if self.point_cloud:
                         mask_i = np.logical_and(laser.sem_label == cls_i, laser.inst_label == id_)
                     else:    
                         mask_i = np.logical_and(laser.proj_sem_label == cls_i, laser.proj_inst_label == id_) 
                     if np.sum(mask_i) < 1:
                         continue
                     lbl_out[mask_i, 1] = int(cls_i)*1000+int(id_)+1
                     lbl_out[mask_i, 0] = iss_class_id 
        np.save(path.join(self.out_dir, label_id + ".bin"), lbl_out)
        return True                    
 
 
def _get_list(base_dir, sub_dirs, cfg):
    _list = []
    for subdir in sub_dirs:
         subdir = '{0:02d}'.format(int(subdir))
         subdir_path = path.join(base_dir, subdir)
         if path.isdir(subdir_path):
             paths = glob.glob(path.join(subdir_path, 'labels', '*.label'))
             paths = sorted(paths) 
             for label in paths:
                _, label = path.split(label)
                label_id = subdir+'_'+label.split('.')[0]
                _list.append((subdir, label_id, cfg))
         
    return _list

def _get_meta(cfg):
    num_stuff = sum(1 for lbl in cfg['instances'] if cfg['instances'][lbl]==False and lbl!=255)
    num_thing = sum(1 for lbl in cfg['instances'] if cfg['instances'][lbl]==True and lbl!=255)
    return num_stuff, num_thing

def _mk_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass

def _init_counter(c):
    global counter
    counter = c

class _Counter:
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val

if __name__ == "__main__":
    main(parser.parse_args())
