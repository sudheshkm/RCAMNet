# utils.py
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

def visualize_prediction(image_path, predictor):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
