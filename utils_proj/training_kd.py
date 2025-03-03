from tqdm import tqdm
import torch
from detection.engine import evaluate
import numpy as np
import pandas as pd
from training import plot_losses

def train_student_model_one_epoch(student, teacher, loader, device, criterion,optimizer, epoch):

    student.train()

    teacher.to(device)
    teacher.train()
    for params in teacher.parameters():
        params.requires_grad = False
    all_losses_dict = []
    all_losses = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets]

        teacher_outputs = teacher(images, targets)
        student_outputs = student(images, targets)

        loss_student = torch.stack(list(student_outputs.values()))
        loss_teacher = torch.stack(list(teacher_outputs.values()))

        stud_dict_append = {k: v.item() for k, v in student_outputs.items()}
        stud_losses = sum(loss for loss in student_outputs.values())
        stud_loss_value = stud_losses.item()
        all_losses_dict.append(stud_dict_append)
        all_losses.append(stud_loss_value)

        # Compute the distillation loss
        loss_distillation = criterion(loss_student, loss_teacher)

        # Backward pass and optimization
        total_loss = loss_distillation + stud_losses
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch+1, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

    losses_values = [all_losses_dict['loss_classifier'].mean(),
                        all_losses_dict['loss_box_reg'].mean(),
                        all_losses_dict['loss_rpn_box_reg'].mean(),
                        all_losses_dict['loss_objectness'].mean()
                        ]
    return losses_values, np.mean(all_losses)

def train_KD(student,teacher, num_epoch, optimizer, train_loader, device, lr_scheduler, criterion):
    distinct_losses_print = []
    losses_mean = []
    for ep in range(num_epoch):
        losses_values, mean = train_student_model_one_epoch(student, teacher, train_loader, device, criterion, optimizer, ep)
        lr_scheduler.step()
        distinct_losses_print.append(losses_values)
        losses_mean.append(mean)
    plot_losses(distinct_losses_print, losses_mean, "Custom_Backbone")
    return student

def train_KD_IoU_one_epoch(student, teacher, loader,val_loader, device, criterion,optimizer, epoch, IoU_teacher):

    student.train()
    teacher.to(device)
    teacher.train()
    for params in teacher.parameters():
        params.requires_grad = False
    all_losses_dict = []
    all_losses = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets]
        student_outputs = student(images, targets)

        stud_dict_append = {k: v.item() for k, v in student_outputs.items()}
        stud_losses = sum(loss for loss in student_outputs.values())
        stud_loss_value = stud_losses.item()
        all_losses_dict.append(stud_dict_append)
        all_losses.append(stud_loss_value)

        coco_evaluator_stud = evaluate(student, val_loader, device)
        IoU_student = coco_evaluator_stud.coco_eval['bbox'].stats[1]
        IoU_student = torch.as_tensor([IoU_student])

        # Compute the distillation loss
        IoU_loss = criterion(IoU_teacher, IoU_student)

        # Backward pass and optimization
        total_loss = stud_losses + IoU_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch+1, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

    losses_values = [all_losses_dict['loss_classifier'].mean(),
                        all_losses_dict['loss_box_reg'].mean(),
                        all_losses_dict['loss_rpn_box_reg'].mean(),
                        all_losses_dict['loss_objectness'].mean()
                        ]
    return losses_values, np.mean(all_losses)

def train_KD_IoU(student,teacher, num_epoch, optimizer, train_loader,val_loader, device, lr_scheduler, criterion):
    distinct_losses_print = []
    losses_mean = []
    coco_evaluator_teach = evaluate(teacher, val_loader, device)
    IoU_teacher = coco_evaluator_teach.coco_eval['bbox'].stats[1]
    IoU_teacher = torch.as_tensor([IoU_teacher])
    for ep in range(num_epoch):
        losses_values, mean = train_KD_IoU_one_epoch(student, teacher, train_loader,val_loader, device, criterion, optimizer, ep,IoU_teacher)
        lr_scheduler.step()
        distinct_losses_print.append(losses_values)
        losses_mean.append(mean)
    plot_losses(distinct_losses_print, losses_mean, "Custom_Backbone")
    return student