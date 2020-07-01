# Panoptic-Tracking
## Evaluation
### System Requirements

#### Programming Language
```
Python 3.6
```

#### Python Packages
```
torch==1.1.0
umsgpack
tqdm
yaml
```

#### Data Preparation
* The prediction and ground truth labels are assumed to be saved with extension '.bin' consisting of 2 channels. The first channel corresponds to the semantic label whereas the second channel corresponds to the tracking ids.
* The tracking id corresponding to stuff predictions are assumed to be 0.
* Each things instance prediction is assumed to have a unique track id for a defined sequence.
* The mapping of stuff and thing classes is assumed to be in the form 0,1,2,...,num_stuff-1 followed by mapping for things segmentation classes. An example of the mapping required can be found in semantic-kitti-ours.yaml file.
* We provide convert_semantic_kitti_gt.py script that converts the SemanticKITTI labels in the required format. To use the evaluation script for custom data write a conversion script similar to the one provided.
```
python convert_semantic_kitti_gt.py  --root_dir path_to_SemanticKITTI/sequences --out_dir path_to_save --config semantic-kitti-ours.yaml --split train or valid --point_cloud true or false
  ```
* --point_cloud, when set to true converts point cloud labels and when set to false converts spherical projections
* Additionally, --depth_cutoff can be used to define depth value after which you don't want to consider instance predictions for evaluation.

#### Running Evalaution
```


python evaluate.py --gt_dir path_to_ground_truths --prediction_dir path_to_predictions
  ```
* MOTSA, sMOTSA, MOTSP, PTQ and sPTQ metric evaluations are saved in a results.json file in the current directory.
