import cv2
from pycocotools.coco import COCO
import os
import copy
import torch
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torchvision.utils import draw_bounding_boxes

def get_transforms(train=False):
        if train:
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.1),
                A.ColorJitter(p=0.1),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco'))
        else:
            transform = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco'))
        return transform

def print_sample(sample, classes, flag = True):
    img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
    color = []
    for i in sample[1]['labels']:
        c = i.item()
        if c == 1:
            color.append((255,0,0))
        else:
            color.append((0,255,0))
    color = list(color)
    img =draw_bounding_boxes( img_int, sample[1]['boxes'], [classes[i-1] for i in sample[1]['labels']], colors = color, width=2,
        font=r"code\calibri.ttf", font_size=20).permute(1, 2, 0)
    if flag:
        plt.imshow(img)
        plt.show()
    else:
        return img

def collate_fn(batch):
        return tuple(zip(*batch))

def get_classes(annot_path):
    coco = COCO(annot_path)
    categories = coco.cats
    n_classes = len(categories.keys())+1
    classes = [i[1]['name'] for i in categories.items()]
    return n_classes, classes

class Dataset(datasets.VisionDataset):
        def __init__(self, root, type = r'train',transform=None, target_transform=None, transforms=None):
            # the 3 transform parameters are reuqired for datasets.VisionDataset
            super().__init__(root, transforms, transform, target_transform)
            self.type = type #train, valid, test
            self.root = root
            self.coco = COCO(os.path.join(self.root,self.type,"annotations.json")) # annotatiosn stored here

            self.ids = list(sorted(self.coco.imgs.keys()))
            self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
            self.classes = ['background', 'player', 'referee']
        def _load_image(self, id: int):
            path = self.coco.loadImgs(id)[0]['file_name']
            image = cv2.imread(os.path.join(self.root,self.type, path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        def _load_target(self, id):
            return self.coco.loadAnns(self.coco.getAnnIds(id))

        def __getitem__(self, index):
            id = self.ids[index]
            image = self._load_image(id)
            target = self._load_target(id)
            target = copy.deepcopy(self._load_target(id))

            boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
            if self.transforms is not None:
                transformed = self.transforms(image=image, bboxes=boxes)

            image = transformed['image']
            boxes = transformed['bboxes']

            new_boxes = [] # convert from xywh to xyxy
            for box in boxes:
                xmin = box[0]
                xmax = xmin + box[2]
                ymin = box[1]
                ymax = ymin + box[3]
                new_boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.tensor(new_boxes, dtype=torch.float32)

            targ = {} # here is our transformed target
            targ['boxes'] = boxes
            targ['labels'] = torch.tensor([t['category_id']  for t in target], dtype=torch.int64)
            targ['image_id'] = torch.tensor([t['image_id'] for t in target])
            targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
            targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
            return image.div(255), targ # scale images
        def __len__(self):
            return len(self.ids)