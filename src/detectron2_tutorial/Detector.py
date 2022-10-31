# flake8: noqa
from operator import mod
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2


class Detector:
    def __init__(self, model_type: str = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        if self.model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif self.model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif self.model_type == "KP":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        else:
            raise ValueError('Unknown model type. Allowed: OD, IS, KP.')

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cuda'

        self.predictor = DefaultPredictor(self.cfg)

    def on_img(self, img_path: str):
        img = cv2.imread(img_path)
        preds = self.predictor(img)

        viz = Visualizer(
            img[:, :, ::-1], instance_mode=ColorMode.IMAGE_BW,
            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        
        output = viz.draw_instance_predictions(preds["instances"].to('cpu'))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)


det = Detector(model_type="KP")
det.on_img("imgs/shepard.jpg")
