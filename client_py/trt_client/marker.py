import cv2


def draw_bboxs(image, bboxs, color=(0, 255, 0), thickness=2):
    cv2.rectangle(
        image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness
    )
