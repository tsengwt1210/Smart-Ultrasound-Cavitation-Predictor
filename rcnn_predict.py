import sys
import torch
import os
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F


def load_model(model_path, device, num_classes=2):
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, device, image_path, threshold=0.6, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return None, []


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)


    with torch.no_grad():
        outputs = model([img_tensor])


    pred = outputs[0]
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    masks = pred['masks'].cpu().numpy()


    keep = (scores >= threshold) & (labels == 1)
    scores = scores[keep]
    masks = masks[keep]


    results = []  # 存 (score, area)
    for mask, score in zip(masks, scores):
        mask_bin = (mask[0] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            area = cv2.contourArea(contour)
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)  # 只畫輪廓
            results.append((score, area))


    if output_path:
        cv2.imwrite(output_path, img)


    return img, results




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python f2_predict.py <model_path> <image_path> <output_path>")
        sys.exit(1)


    modelPath = sys.argv[1]
    imgPath   = sys.argv[2]
    outPath   = sys.argv[3]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(modelPath, device)
    _, results = predict_image(model, device, imgPath, threshold=0.6, output_path=outPath)


    # 印出 score 與像素面積
    for i, (score, area) in enumerate(results, 1):
        print(f"物件{i}: Score={score:.2f}, PixelArea={area:.0f}")





