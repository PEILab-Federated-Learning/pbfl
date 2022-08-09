import os
import torch
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import transforms
from dataloader.base import Datum
from dataloader.utils.tools import mkdir_if_missing, read_image
from model.clip import load_clip_to_cpu
from model.prompt import CustomCLIP


from yacs.config import CfgNode as CN
_C = CN()
_C.VERBOSE = False
_C.USE_CUDA = True
_C.OUTPUT_DIR = "./dlg_output"
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = "../../dataset"
_C.DATASET.NAME = "imagenet"
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "RN50"
_C.PROMPT = CN()
_C.PROMPT.N_CTX = 16  # number of context vectors
_C.PROMPT.CTX_INIT = ""  # initialization words
_C.PROMPT.PREC = "fp16"  # fp16, fp32, amp
_C.PROMPT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


class model:
    def __init__(self, cfg, classnames):
        self.cfg = cfg
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.output_dir = cfg.OUTPUT_DIR
        self.classnames = classnames
        self.clip_model = self.load_clip()
        

    def load_clip(self):
        cfg = self.cfg
        if cfg.VERBOSE:
            print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.PROMPT.PREC == "fp32" or cfg.PROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        return clip_model
    
    
    def init_model(self):
        cfg = self.cfg
        classnames = self.classnames
        clip_model = self.clip_model
        # Building custom CLIP
        model = CustomCLIP(cfg, classnames, clip_model)
        # Turning off gradients in both the image and the text encoder
        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        model.to(self.device)
        return model


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_classnames(text_file):
    """Return a dictionary containing
       key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames 


def read_data(image_dir, classnames, split_dir):
    split_dir = os.path.join(image_dir, split_dir)
    folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
    items = []
    for label, folder in enumerate(folders):
        imnames = listdir_nohidden(os.path.join(split_dir, folder))
        classname = classnames[folder]
        for imname in imnames:
            impath = os.path.join(split_dir, folder, imname)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)
        return items
    

def imagenet_loader():
    dataset_dir = "imagenet"
    cfg = _C
    root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
    dataset_dir = os.path.join(root, dataset_dir)
    image_dir = os.path.join(dataset_dir, "images")
    cls_file = os.path.join(dataset_dir, "classnames.txt")
    classnames = read_classnames(cls_file)
    val = read_data(image_dir, classnames, "val")
    return classnames, val


def run_dlg():
    cfg = _C
    
    output_dir = os.path.abspath(os.path.expanduser(cfg.OUTPUT_DIR))
    mkdir_if_missing(output_dir)
    
    classnames, data_source = imagenet_loader()
    
    _model = model(cfg, classnames)
    prompt_model = _model.init_model()
    prompt_learner = prompt_model.prompt_learner
    
    tp = transforms.Compose([transforms.Resize(cfg.INPUT.SIZE),
                             transforms.ToTensor(),])
    tt = transforms.ToPILImage()
    
    img_index = 10
    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    item = data_source[img_index]
    img0 = read_image(item.impath)
    gt_data = tp(img0).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([item.label]).long().to(device)
    gt_label = gt_label.view(1, )
    
    #print(gt_data.shape)
    #print(gt_label.shape)
    
    #plt.imshow(tt(gt_data[0].cpu()))
    
    # compute original gradient 
    pred = prompt_model(gt_data)
    
    y = F.cross_entropy(pred, gt_label)
    dy_dx = torch.autograd.grad(y, prompt_learner.parameters())
    
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    
    # generate dummy data 
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    
    #plt.imshow(tt(dummy_data[0].cpu()))
    
    optimizer = torch.optim.LBFGS([dummy_data])

    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = prompt_model(dummy_data) 
            dummy_loss = F.cross_entropy(dummy_pred, gt_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, 
                                              prompt_learner.parameters(), 
                                              create_graph=True)

        
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
        
            return grad_diff
        
        
        optimizer.step(closure)
        return
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    plt.show()
    

if __name__ == "__main__":
    run_dlg()
    