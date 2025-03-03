import math
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from detection.engine import evaluate


def train_one_epoch(model, optimizer, loader, device, epoch):
        model.to(device)
        model.train()
        all_losses = []
        all_losses_dict = []

        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
            losses = sum(loss for loss in loss_dict.values()) #sum losses of this dataloader
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()

            all_losses.append(loss_value)
            all_losses_dict.append(loss_dict_append)

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training") # train if loss becomes infinity
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
        print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch+1, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))
        '''
        Region Proposal Network is a subcomponent of the Fast RCNN and Faster RCNN architectures. It proposes candidate boxes and scores whether there is an object in this regions.
        RPN loss and objectness loss must be losses of this predictions.
        Regressor loss is the loss of the prediction of bounding box coordinates,
        and classifier loss is the loss of prediction of object classes in bounding boxes.
        IOU is acronym for intersection over union, and it gives how much bounding boxes are overlapped. In RPN it is calculated between suggested boxes(anchors), and ground truths. Higher IOU scores means suggested box includes an object of interest.
        Average precision is the average of the areas under the precision recall curve for each of the object classes.
        '''
        losses_values = [all_losses_dict['loss_classifier'].mean(),
                        all_losses_dict['loss_box_reg'].mean(),
                        all_losses_dict['loss_rpn_box_reg'].mean(),
                        all_losses_dict['loss_objectness'].mean()
                        ]
        return losses_values, np.mean(all_losses)

def train_new_model(model, num_epoch, optimizer, train_loader, device, lr_scheduler, backbone):
    distinct_losses_print = []
    losses_mean = []
    for ep in range(num_epoch):
        losses_values, mean = train_one_epoch(model, optimizer, train_loader, device, ep)
        lr_scheduler.step()
        distinct_losses_print.append(losses_values)
        losses_mean.append(mean)
    #plot_losses(distinct_losses_print, losses_mean, backbone)
    return model

def train_few_shot(model, num_epoch, optimizer, train_loader,test_loader, device, lr_scheduler,warmup_scheduler, backbone):
    distinct_losses_print = []
    losses_mean = []
    mAP = []
    for ep in range(num_epoch):
        losses_values, mean = train_one_epoch(model, optimizer, train_loader, device, ep)
        warmup_scheduler.step()
        if ep == 4:
            warmup_scheduler = lr_scheduler
        distinct_losses_print.append(losses_values)
        losses_mean.append(mean)
        coco_evaluator = evaluate(model, test_loader, device)
        mAP.append(coco_evaluator.coco_eval['bbox'].stats[1])
    plot_precision(mAP, backbone+" few_shot")
    plot_losses(distinct_losses_print, losses_mean, backbone + " few_shot")
    return model

def train_and_test(model, num_epoch, optimizer, train_loader, test_loader,device, lr_scheduler, backbone):
    distinct_losses_print = []
    losses_mean = []
    mAP = []
    for ep in range(num_epoch):
        losses_values, mean = train_one_epoch(model, optimizer, train_loader, device, ep)
        lr_scheduler.step()
        distinct_losses_print.append(losses_values)
        losses_mean.append(mean)
        coco_evaluator = evaluate(model, test_loader, device)
        mAP.append(coco_evaluator.coco_eval['bbox'].stats[1])
    plot_precision(mAP, backbone)
    plot_losses(distinct_losses_print, losses_mean, backbone)
    return model

def plot_losses(distinct_losses, losses_mean, backbone):
    distinct_losses = np.array(distinct_losses)
    fig = plt.figure()
    plt.plot(distinct_losses[:,0], label='loss_classifier')
    plt.plot(distinct_losses[:,1], label='loss_box_reg')
    plt.plot(distinct_losses[:,2], label='loss_rpn_box_reg')
    plt.plot(distinct_losses[:,3], label='loss_objectness')
    plt.plot(losses_mean, label = "losses_sum")
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend(fontsize="7",loc="upper right")
    plt.title('FasterRCNN - {} - {} epochs'.format(backbone,len(losses_mean)))
    fig.savefig("code/plots/{}_{}ep_losses.png".format(backbone,len(losses_mean)))

def plot_precision(precision, backbone):
    precision = np.array(precision)
    fig = plt.figure()
    plt.plot(precision)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('FasterRCNN - {} - {} epochs with IoU > 0.5'.format(backbone, len(precision)))
    fig.savefig("code/plots/{}_{}ep_precision.png".format(backbone,len(precision)))

