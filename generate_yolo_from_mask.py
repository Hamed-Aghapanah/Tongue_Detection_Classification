import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import tqdm
import os

def load_mask(path):
    mask = cv2.imread(path)
    # mask[:, :, 0] = (mask[:, :, 0] > 0) * 255
    # mask[:, :, 1] = (mask[:, :, 1] > 0) * 255
    # mask[:, :, 2] = (mask[:, :, 2] > 0) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # mask = (mask > 100) * 255
        
    return mask

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

datapath = r"F:\project\Tehran\2PROGNOSMART\mfiles\000_Tongue_segmentation_dataset\masks"
for image_path in tqdm.tqdm(glob.glob(datapath + "/*")):
    try:
        name = image_path.split(os.path.sep)[-1].split(".")[0]    
        mask = load_mask(image_path)
        mask = np.uint8(mask)
        h_mask, w_mask = mask.shape    
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x,y,w,h = cv2.boundingRect(cnt)
        out = xml_to_yolo_bbox([x, y, x + w, y + h], w_mask, h_mask)
        with open("txt/" + name + ".txt", "w") as f:
            string = "0 " + str(out[0]) + " " + str(out[1]) + " " + str(out[2]) + " " + str(out[3])
            f.write(string)
    except:
        pass

    
# plt.imshow(mask)
# plt.show()