import torch
import os
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes
from PIL import Image

sys.path.append(r"code\utils_proj\dataset")
import utils_proj.dataset as ds


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_custom_image(path, model, classes):
    test_img = Image.open(path)
    test_img = np.array(test_img)
    transform = A.Compose([ToTensorV2()])
    test_test = transform(image = test_img)
    with torch.no_grad():
        start = time.time()
        prediction = model(((test_test['image'])/255).unsqueeze(0))
        end = time.time()
        print("Inference time: {:.2f} seconds".format(end-start))
        pred = prediction[0]
    color = []
    for i in pred['labels']:
        c = i.item()
        if c == 1:
            color.append((255,0,0))
        else:
            color.append((0,255,0))
    color = list(color)
    print(pred['scores'])
    plt.imshow(draw_bounding_boxes(test_test['image'],
        pred['boxes'][pred['scores'] > 0.8],
        list([classes[i-1]+ "  "+ str(round(pred['scores'][j].item(), 2)) for j,i in enumerate(pred['labels'][pred['scores'] > 0.8])]), colors=color,width=2,
     font = r"code\calibri.ttf", font_size=25).permute(1, 2, 0))
    # plt.show()
    name = "code/output/" +  path.split('//'[0])[-1]

    plt.savefig(name)

def testing_validationset(valset, model, classes):
    start = time.time()
    print("Starting validation...")
    for sample in valset:

        '''fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1,2,1)
        gt = ds.print_sample(sample, classes, flag = False)
        plt.imshow(gt)
        plt.title("Ground Truth")
        fig.add_subplot(1,2,2)'''
        sample = np.array(sample)
        #img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
        with torch.no_grad():
            prediction = model(sample[0])
            '''pred = prediction[0]
        color = []
        for i in pred['labels']:
            c = i.item()
            if c == 1:
                color.append((255,0,0))
            else:
                color.append((0,255,0))
        color = list(color)
        print(pred['scores'])
        plt.imshow(draw_bounding_boxes(img_int,
            pred['boxes'][pred['scores'] > 0.7],
            list([classes[i-1] + "  "+ str(round(pred['scores'][j].item(), 2)) for j,i in enumerate(pred['labels'][pred['scores'] > 0.7])]), colors=color,width=2,
            font = r"code\calibri.ttf", font_size=15).permute(1, 2, 0))
        plt.title("Prediction")
        plt.show()
        #plt.savefig(name)'''
    end = time.time()
    print("Validation time : {:.2f} seconds".format(end-start))