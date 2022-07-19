import os.path as osp
from collections import OrderedDict
import pickle
from functools import partial
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from .clip import load_clip_to_cpu
from .prompt import CustomCLIP
from .optimizers.optimizers import build_optimizer
from .schedulers.lr_schedulers import build_lr_scheduler

                

class Model:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.dataset = dataset
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
    
    
    def load_checkpoint(self, fpath):
        r"""Load checkpoint.
        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.
        Args:
            fpath (str): path to checkpoint.
        Returns:
            dict
        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError("File path is None")
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        map_location = None if torch.cuda.is_available() else "cpu"
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint
    
    
    def load_model_weights(self, model, weight_path):
        cfg = self.cfg
        if not weight_path:
            if cfg.VERBOSE:
                print("Note that load_model() is skipped as no model is given")
            return

        if not osp.exists(weight_path):
            raise FileNotFoundError('Model not found at "{}"'.format(weight_path))
        checkpoint = self.load_checkpoint(weight_path)
        state_dict = checkpoint["state_dict"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]
        if cfg.VERBOSE:
            print("Loading model weights from {}".format(weight_path))
        # set strict=False
        model.load_state_dict(state_dict, strict=False)
    
    
    def save_checkpoint(self, state, save_dir, remove_module_from_keys=True):
        r"""Save checkpoint.
        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is True.
        Examples::
            >>> state = {
            >>>     'state_dict': model.state_dict(),
            >>>     'epoch': 10,
            >>>     'optimizer': optimizer.state_dict()
            >>> }
            >>> save_checkpoint(state, 'log/my_model')
        """
        cfg = self.cfg

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        torch.save(state, save_dir)
        if cfg.VERBOSE:
            print(f"Checkpoint saved to {save_dir}")
    
    
    def save_model(self, model, directory):
        model_dict = model.state_dict()
        self.save_checkpoint({"state_dict": model_dict}, directory)


    def build_model(self, weight_path):
        cfg = self.cfg
        classnames = self.dataset.classnames
        clip_model = self.clip_model
        # Building custom CLIP
        model = CustomCLIP(cfg, classnames, clip_model)

        # Turning off gradients in both the image and the text encoder
        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.load_model_weights(model.prompt_learner, weight_path)
        
        model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        optim = build_optimizer(model.prompt_learner, cfg.OPTIM)
        sched = build_lr_scheduler(optim, cfg.OPTIM)
        scaler = GradScaler() if cfg.PROMPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            if cfg.VERBOSE:
                print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            model = nn.DataParallel(model)
            
        return model.prompt_learner, optim, sched, scaler
    
    
    def init_model(self, save_path):
        cfg = self.cfg
        classnames = self.dataset.classnames
        clip_model = self.clip_model
        # Building custom CLIP
        model = CustomCLIP(cfg, classnames, clip_model)
    
        self.save_model(model.prompt_learner, save_path)
        
    