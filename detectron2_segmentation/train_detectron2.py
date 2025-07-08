# train_detectron2.py

import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
import os

# register dataset
register_coco_instances("blb_train", {}, "datasets/train/annotations.json", "datasets/train/images")
register_coco_instances("blb_val", {}, "datasets/val/annotations.json", "datasets/val/images")

cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.DATASETS.TRAIN = ("blb_train",)
cfg.DATASETS.TEST = ("blb_val",)
cfg.OUTPUT_DIR = "./output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
