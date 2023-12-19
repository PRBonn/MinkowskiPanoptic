Panoptic segmentation network using MinkowskiNet as backbone and predicting offset and clustering

# Installation

Install requirements

Install this repo as a package via 

```
pip3 install -U -e .
```

# how to run

Training

```
python3 scripts/train_model.py


Options:
  --w TEXT            weights to load
  --ckpt [ckpt_path]  checkpoint to resume training
  --nuscenes
  --mini              use mini split for nuscenes
  --seq [seq_num]     use a single sequence for train and val
  --id [exp_name]    set id of the experiment
  --help             Show this message and exit.
```

Evaluation

```
python3 scripts/evaluate_model.py --w path_to_weights


Options:
  --w [path_to_weights]   weights to load
  --save                  save ply predictions
  --save_testset          save predictions for test set
  --nuscenes              use nuscenes dataset
  --mini                  use mini split for nuscenes
  --seq [seq_num]         use a single sequence for train and val
  --help                  Show this message and exit.
```
