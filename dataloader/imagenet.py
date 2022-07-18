import os
import torch
import pickle
import random
from collections import OrderedDict, defaultdict
from .utils.tools import mkdir_if_missing, listdir_nohidden
from .base import Datum, DatasetWrapper
from .transforms.transforms import build_transform
from .samplers.samplers import build_sampler


class ImageNet:
    dataset_dir = "imagenet"  # the directory where the dataset is stored
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        """Create dirname if it is missing."""
        mkdir_if_missing(self.split_fewshot_dir)
        
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed_rawdata = pickle.load(f)
                train = preprocessed_rawdata["train"]
                self.test = preprocessed_rawdata["test"]
        else:
            cls_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(cls_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            self.test = self.read_data(classnames, "val")
            preprocessed_rawdata = {"train": train, "test": self.test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed_rawdata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.num_shots = cfg.DATASET.NUM_SHOTS
        self.seed = cfg.SEED
        self.preprocessed_shot = os.path.join(self.split_fewshot_dir, f"shot_{self.num_shots}-seed_{self.seed}.pkl")
        if os.path.exists(self.preprocessed_shot):
            if cfg.VERBOSE:
                print(f"Loading preprocessed few-shot data from {self.preprocessed_shot}")
            with open(self.preprocessed_shot, "rb") as f:
                preprocessed_shotdata = pickle.load(f)
                self.train_shot = preprocessed_shotdata["train_shot"]
        else:   
            self.train_shot = self.generate_fewshot_dataset(train, num_shots=self.num_shots)
            preprocessed_shotdata = {"train_shot": self.train_shot}
            if cfg.VERBOSE:
                print(f"Saving preprocessed few-shot data to {self.preprocessed_shot}")
            with open(self.preprocessed_shot, "wb") as f:
                pickle.dump(preprocessed_shotdata, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        self.tfm_train = build_transform(cfg, is_train=True)
        self.tfm_test = build_transform(cfg, is_train=False)
        
        ''' extremely non-IID data partition, Kolmogorov–Smirnov(KS)=1 '''
        clients_total = cfg.FL.CLIENTS_TOTAL
        samples_per_label = cfg.DATASET.NUM_SHOTS
        tracker = self.split_dataset_by_label(self.train_shot)
        self.num_classes = len(tracker)
        labels_per_client = int(self.num_classes/clients_total)
        samples_per_client = labels_per_client*samples_per_label
        self.trainset = []
        client_trainset_tmp = []
        for label, items in tracker.items():
            client_trainset_tmp.extend(items)
            if(len(client_trainset_tmp)==samples_per_client):
                self.trainset.append(client_trainset_tmp)
                client_trainset_tmp = []
            
        self.lab2cname, self.classnames = self.get_lab2cname(self.train_shot)
    
    
    def build_data_loader(self, client_id=-1, is_train=True):
        cfg = self.cfg
        # Build sampler
        if is_train:
            sampler_type = cfg.DATALOADER.TRAIN_X.SAMPLER
            data_source = self.data_partition(client_id)
        else:
            sampler_type = cfg.DATALOADER.TEST.SAMPLER
            data_source = self.test
        sampler = build_sampler(sampler_type, data_source=data_source)

        # Build data loader
        if is_train:
            data_loader = torch.utils.data.DataLoader(
                DatasetWrapper(cfg, data_source, self.tfm_train, is_train=is_train),
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=len(data_source) >= cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                DatasetWrapper(cfg, data_source, self.tfm_test, is_train=is_train),
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
            )
        assert len(data_loader) > 0

        return data_loader
    
    
    ''' extremely non-IID data partition, Kolmogorov–Smirnov(KS)=1 '''
    def data_partition(self, client_id):
        cfg = self.cfg
        if cfg.VERBOSE:
            print('Data partition is done for client', client_id+1, '...')
        return self.trainset[client_id]
        

    @staticmethod
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
    
    
    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
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
    
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        """Generate a few-shot dataset (typically for the training set).
        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        cfg = self.cfg
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources
        if cfg.VERBOSE:
            print(f"Creating a {num_shots}-shot dataset")

        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output
    
    
    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.
        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)

        return output


    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).
        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
        