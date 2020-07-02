# MOPT: Multi-Object Panoptic Tracking

## Official implementation for "Multi-Object Panoptic Tracking"  <br/>[[Paper](https://arxiv.org/abs/2004.08189)] 

[Juana Valeria Hurtado ](http://rl.uni-freiburg.de/people/hurtado), [Rohit Mohan](http://rl.uni-freiburg.de/people), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard/), and [Abhinav Valada](http://rl.uni-freiburg.de/people/valada).


## System Requirements

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


## Evaluation Metric
![This is the caption\label{track}](https://github.com/robot-learning-freiburg/Panoptic-Tracking/blob/master/track_metric.png)

Our paper proposes two metrics for evaluating performance of Panoptic Tracking: PTQ and sPTQ (soft version of PTQ).<br>

The figure above depicts computation of PTQ for a class c. Here, class c is the Car class. The figure shows ground truth of two consecutive frames as well as the prediction of an arbitrary panoptic tracking model. In ground truth three instances of car is present: Car-1, Car-3 and Car-4 and one instance of van: Van-1. In prediction for both frames 3 instances of Car is observed. Now, when considering the Car class, Car-1(x2), Car-3 and Car-10 prediction corresponds to true  positive (TP<sub>c</sub>). Car-2(x2) prediction contributes to false positive (FP<sub>c</sub>) and Void prediction for Car-4 contributes to false negative (FN<sub>c</sub>). Lastly, the switch between Car-3 to Car-10 in the predictions shows ID switching (IDS<sub>c</sub>). Once individual terms are computed PTQ<sub>c</sub> can be obtained using the above mentioned formula. The IDS<sub>c</sub> is observed only for things classes (classes having instances) and is 0 in case of stuff classes.  

### Training
    Coming Soon !!!
### Testing 
    Coming Soon !!!
    
### Evaluation 

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


## Citation

If you find the code useful for your research, please consider citing our paper:

```bibtex
@article{hurtado2020mopt,
  title={MOPT: Multi-Object Panoptic Tracking},
  author={Hurtado, Juana Valeria and Mohan, Rohit and Valada, Abhinav},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshop on Scalability in Autonomous Driving},
  year={2020}
}
}
```

## License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.


## Acknowledgements
This project has used utility functions from other wonderful open-sourced libraries. We would especially thank the authors of:
* [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
* [seamseg](https://github.com/mapillary/seamseg)

## Contact

If you have any questions regarding the repo, please contact 
Juana Valeria Hurtado (hurtadoj@informatik.uni-freiburg.de) or create an issue.

