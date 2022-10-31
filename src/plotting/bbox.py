import cv2
import numpy as np
from typing import Tuple, List, Dict, Union
import colorsys

BBOX_POS = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BB_THICKNESS = 4
PADDING = int(BB_THICKNESS/2)


def _get_label_box_pos(text, box):
    (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, FONT_SCALE, FONT_THICKNESS)
    label_box = [
        box[0]-PADDING, box[1]-text_height-PADDING,
        box[0]+text_width+PADDING, box[1]+PADDING]
    if label_box[1] < 0:
        label_box[1] = 0
        label_box[3] = text_height
    return label_box


def _add_label(
    img: np.ndarray, label: str,
    box: BBOX_POS, color: RGB
) -> np.ndarray:
    label_box = _get_label_box_pos(label, box)
    cv2.rectangle(img, label_box[:2], label_box[2:4], color, -1)

    cv2.putText(
        img, label, (label_box[0]+PADDING, label_box[3]-PADDING), FONT,
        FONT_SCALE, color=(255, 255, 255), thickness=FONT_THICKNESS)
    return img


def _create_palette(label_ids: List[int]):
    len_ = len(set(label_ids))
    if len_ % 2 == 0:
        len_ += 1
    labels_palette = [
        colorsys.hsv_to_rgb(((2 * i % len_) / len_), 1, 0.5)
        for i in range(len_)]
    labels_palette = {
        label: (int(r*255), int(g*255), int(b*255))
        for label, (r, g, b) in zip(set(label_ids), labels_palette)}
    return labels_palette


def _set_palette(
    labels: List[str], labels_palette: Union[List[RGB], Dict[str, RGB]] = None
) -> Dict[str, RGB]:
    if labels_palette is None:
        labels_palette = _create_palette(labels)
    elif isinstance(labels_palette, List) or isinstance(labels_palette, Tuple):
        labels_set = set(labels_palette)
        assert len(labels_set) <= len(labels_palette),\
            "More labels than colors."
        labels_palette = {
            label: lp for label, lp in zip(
                labels, labels_palette[:len(labels)])
        }
    return labels_palette


def add_labeled_bbox(
    img: np.ndarray, scores: List[float], label_ids: List[int],
    boxes: List[BBOX_POS], labels_dict: Dict[int, str], th: float = 0.9,
    labels_palette: Union[List[RGB], Dict[int, RGB]] = None
) -> np.ndarray:
    labels_palette = _set_palette(label_ids, labels_palette)
    for score, label_id, box in zip(scores, label_ids, boxes):
        if score < th:
            continue
        box = [int(round(val)) for val in box]
        label_text = labels_dict[label_id]
        color = labels_palette[label_id]

        cv2.rectangle(img, box[0:2], box[2:4], color, BB_THICKNESS)
        _add_label(img, label_text, box, color)
    return img
