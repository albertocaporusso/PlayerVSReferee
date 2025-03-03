import torch
import sys
import os

from torch.backends.quantized import supported_engines
from torch import optim
from torch.utils.data import DataLoader
from utils_proj import *
sys.path.append(r"code\detection")
from detection.engine import evaluate


def main():



    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    DATA_PATH = "dataset"
    ANNOT_PATH = "dataset/train/annotations.json"
    BACKBONE = "resnet50" #mobilenet or resnet50
    #GET PATHS and CLASSES
    n_classes, classes = ds.get_classes(annot_path=ANNOT_PATH)


    #instantiate sets
    #train_dataset = ds.Dataset(root=DATA_PATH, type = 'train',transforms=ds.get_transforms(True))
    #test_dataset = ds.Dataset(root=DATA_PATH, type = 'test',transforms=ds.get_transforms(False))
    validation_set = ds.Dataset(root=DATA_PATH, type = 'validation', transforms=ds.get_transforms(False))
    #fewshot_set = ds.Dataset(root=DATA_PATH, type = 'fewshot',transforms=ds.get_transforms(True))
    #smaller_fewshot_set = ds.Dataset(root=DATA_PATH, type = 'smaller_fewshot',transforms=ds.get_transforms(True))


    # THE DATA LOADERS
    #train_loader = DataLoader(dataset = train_dataset, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
    #test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False, collate_fn=ds.collate_fn)
    val_loader = DataLoader(dataset = validation_set, batch_size=1, shuffle=True, collate_fn=ds.collate_fn)
    #few_shot_loader = DataLoader(dataset =fewshot_set, batch_size=3, shuffle=True, collate_fn=ds.collate_fn)
    #smaller_few_shot_loader = DataLoader(dataset =smaller_fewshot_set, batch_size=1, shuffle=True, collate_fn=ds.collate_fn)
    #GET A SAMPLE from training set and show
    #ds.print_sample(train_dataset[5],classes)
    pruning_ratio = 0.8
    model = pr.global_pruning(BACKBONE,pruning_ratio)
    #torch.save(model, 'code/models/pruned_{}_{}.pth'.format(BACKBONE, str(pruning_ratio)[0]+str(pruning_ratio)[2]))
    #model = torch.load('code/models/resnet50/model_20ep.pth',map_location='cpu')
    #evaluate(model, test_loader, 'cpu')
    model.eval()
    ts.testing_validationset(val_loader,model, classes)
    exit()




    # THE MODEL TO BE TRAINED Quant
    '''model = md.get_model(backbone = BACKBONE, n_classes = n_classes)
    model.quant = quantization.QuantStub()
    model.dequant = quantization.DeQuantStub()
    #print(supported_engines)
    model.eval()
    model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
    model.train()
    model = quantization.prepare_qat(model, inplace=True)

    #OPTIMIZER AND LEARNING RATE
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.005, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    warm_up_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch + 1) / 10, 1))
    criterion = CrossEntropyLoss()

    ##TRAINING Quant
    num_epochs=20
    model = tr.train_new_model(model, num_epochs, optimizer, train_loader, DEVICE, lr_scheduler, BACKBONE)
    quantized_model.eval()
    quantized_model = quantization.convert(model, inplace=True)
    evaluate(quantized_model, test_loader,DEVICE)
    #model = tr.train_KD(model,num_epochs,optimizer,train_loader,DEVICE,lr_scheduler,warm_up_scheduler, criterion, n_classes)
    #TO SAVE MODEL (N.B. the name has to be 'name.pth')
    torch.save(quantized_model, os.path.join(r"code\models",BACKBONE,'model_quantized_20ep.pth'))'''

    '''##TRAINING FOR KNOWLEDGE DISTILLATION
    teacher = torch.load('code/models/resnet50/model_20ep.pth', map_location=torch.device('cpu'))
    num_epochs = 30

    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.005, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    warm_up_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch + 1) / 10, 1))
    criterion = CrossEntropyLoss()

    torch.save(student_trained, 'code/student_{}ep'.format(num_epochs))'''

    #TO LOAD PRETRAINED MODEL
    '''model_trained = torch.load('code/models/model_fs_15ep.pth', map_location= torch.device('cpu'))

    model_trained.eval()
    ts.testing_validationset(validation_set, model_trained, classes)
    #print(model_trained)'''

    #FEW SHOT LEARNING we use the validation set as training set in this case

    '''model = md.get_model(backbone = BACKBONE, n_classes = n_classes)
    #model.backbone.requires_grad_(False)
    num_epoch = 20
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum = 0.9, weight_decay=0.0005)
    optimizer = optim.Adam(params, lr=0.005, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    warm_up_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch + 1) / 10, 1))

    model = tr.train_and_test(model, num_epoch, optimizer, train_loader,test_loader, DEVICE, lr_scheduler, BACKBONE)
    torch.save(model, '/content/code/model_fs_20ep.pth')'''
    #EVALUATION OF THE MODEL with image from internet (the images has to be put in "input" folder and the results will be saved in the output folder )
    #ts.test_custom_image("code/input/test_1.jpg",model_trained, classes)

    #EVALUTATION OF THE MODEL with whole testing set
    model_trained = torch.load('code/models/mobilenet/model_20ep.pth', map_location= torch.device('cpu'))

    coco_ev = evaluate(model_trained, test_loader,DEVICE)
    ''' Average Precision (AP) @[IoU=0.50:0.95 | area=all | maxDets=100]: This metric measures the average precision across different IoU thresholds ranging from 0.50 to 0.95. It considers all object sizes (area=all) and allows a maximum of 100 detections per image (maxDets=100). The value of 0.729 indicates that the average precision is 72.9%.
        Average Precision (AP) @[IoU=0.50 | area=all | maxDets=100]: This metric calculates the average precision at a specific IoU threshold of 0.50. It considers all object sizes (area=all) and allows a maximum of 100 detections per image. The value of 0.940 indicates an average precision of 94.0% at an IoU of 0.50.
        Average Precision (AP) @[IoU=0.75 | area=all | maxDets=100]: Similar to the previous metric, this measures the average precision at a higher IoU threshold of 0.75. The value of 0.840 indicates an average precision of 84.0% at an IoU of 0.75.
        Average Precision (AP) @[IoU=0.50:0.95 | area=small | maxDets=100]: This metric focuses on objects with small sizes (area=small). It calculates the average precision across various IoU thresholds from 0.50 to 0.95. The value of 0.320 indicates an average precision of 32.0% for small objects.
        Average Precision (AP) @[IoU=0.50:0.95 | area=medium | maxDets=100]: This metric measures the average precision for objects with medium sizes (area=medium) across the same range of IoU thresholds. The value of 0.453 indicates an average precision of 45.3% for medium objects.
        Average Precision (AP) @[IoU=0.50:0.95 | area=large | maxDets=100]: This metric focuses on objects with large sizes (area=large) and calculates the average precision across the range of IoU thresholds. The value of 0.777 indicates an average precision of 77.7% for large objects.
        Average Recall (AR) @[IoU=0.50:0.95 | area=all | maxDets=1]: This metric measures the average recall (sensitivity) at different IoU thresholds for all object sizes (area=all). It allows only one detection per image (maxDets=1). The value of 0.615 indicates an average recall of 61.5% at IoU thresholds from 0.50 to 0.95.
        Average Recall (AR) @[IoU=0.50:0.95 | area=all | maxDets=10]: Similar to the previous metric, this calculates the average recall, but it allows up to ten detections per image. The value of 0.784 indicates an average recall of 78.4% at IoU thresholds from 0.50 to 0.95.
        Average Recall (AR) @[IoU=0.50:0.95 | area=all | maxDets=100]: This metric measures the average recall with a maximum of 100 detections per image. The value of 0.784 indicates an average recall of 78.4% at'''

    #EVALUATION OF THE MODEL with whole validation set
    #ts.testing_validationset(validation_set, model_trained, classes)

if __name__ == '__main__':
    main()