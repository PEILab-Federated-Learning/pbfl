import os
import copy
import torch
from configs.imagenet_vitb16 import _C
from dataloader.imagenet import ImageNet
from model.model import Model
from dataloader.utils.tools import mkdir_if_missing


class FL:
    clients_dataloader = []
    clients_model = []
    
    def extract_client_updates(self):
        # Extract baseline model weights
        baseline_weights = self.learner_base
        
        # Extract weights from clients' trained models 
        weights = self.clients_model
        
        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]
                # Ensure correct weight is being updated
                assert name == bl_name
                
                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates
    
    def federated_averaging(self):
        cfg = self.cfg
        weight_client = 1 / cfg.FL.CLIENTS_TOTAL
        
        # Extract updates from reports
        updates = self.extract_client_updates()

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            for j, (_, delta) in enumerate(update):
                # Use weighted average 
                avg_update[j] += delta * weight_client

        # Extract baseline model weights
        baseline_weights = self.learner_base

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights
    
    def load_weights(self, model, weights):
        updated_state_dict = {}
        for name, weight in weights:
            updated_state_dict[name] = weight

        model.load_state_dict(updated_state_dict, strict=False)

    
    def set_cfg(self):
        self.cfg = _C.clone()

    def before_train(self):
        print("before_train starting ...")
        
        self.set_cfg()
        cfg = self.cfg
        if cfg.VERBOSE:
            print("********************")
            print("****** Config ******")
            print("********************")
            print(cfg)
        self.seed = cfg.SEED
        self.output_dir = os.path.abspath(os.path.expanduser(cfg.OUTPUT_DIR))
        print("configs done ...")
        
        self.dataset = ImageNet(cfg)
        for client_id in range(cfg.FL.CLIENTS_TOTAL):
            self.clients_dataloader.append(self.dataset.build_data_loader(client_id))
        print("dataset done ...")
            
        self.model = Model(cfg, self.dataset.classnames)
        print("model (clip) done ...")
        
        self.weight_dir = os.path.join(self.output_dir, cfg.MODEL.WEIGHT_PATH)
        mkdir_if_missing(self.weight_dir)
        init_weights = os.path.join(self.weight_dir, f"model_round0_seed{self.seed}.pth.tar")
        self.model.init_model(init_weights)
        print("init weights done ...")

    def run_train(self):
        print("run_train starting ...")
        cfg = self.cfg
        for round_id in range(cfg.FL.ROUNDS_TOTAL):
            print('Starting fl round', '['+str(round_id+1)+'/'+str(cfg.FL.ROUNDS_TOTAL)+']', '...')
            self.before_round(round_id)
            self.run_round(round_id)
            self.after_round(round_id)
            

    def after_train(self):
        print("All training done ...")



    def before_round(self, round_id):
        self.clients_model = []
        
    def after_round(self, round_id):
        cfg = self.cfg
        assert len(self.clients_model) == cfg.FL.CLIENTS_TOTAL
        updated_weights = self.federated_averaging()
        self.load_weights(self.base_model, updated_weights)
        
        weight_file = os.path.join(self.weight_dir, f"model_round{round_id+1}_seed{self.seed}.pth.tar")
        self.model.save_model(self.base_model, weight_file)
        print("round", str(round_id+1), "global update done ...")
        
    
    def run_round(self, round_id):
        cfg = self.cfg
        learner_based_required = True
        for client_id in range(cfg.FL.CLIENTS_TOTAL):
            print('Starting fl client', '['+str(client_id+1)+'/'+str(cfg.FL.CLIENTS_TOTAL)+']', '...')
            
            client_dataloader = self.clients_dataloader[client_id]
            print("client", str(client_id+1), "dataset done ...")
            
            weight_file = os.path.join(self.weight_dir, f"model_round{round_id}_seed{self.seed}.pth.tar")
            model, optim, sched, scaler = self.model.build_model(weight_file)
            if learner_based_required:
                self.base_model = copy.deepcopy(model.prompt_learner)
                self.learner_base = self.model.extract_weights(self.base_model)
                learner_based_required = False
            print("client", str(client_id+1), "model done ...")

            model = self.model.train_model(client_dataloader, model, optim, sched, scaler)
            learner_trained = self.model.extract_weights(model.prompt_learner)
            print("client", str(client_id+1), "train done ...")
            self.clients_model.append(learner_trained)
            
    
    def test(self, eval_only):
        print("test starting ...")
        if eval_only:
            self.set_cfg()
            cfg = self.cfg
            self.seed = cfg.SEED
            self.output_dir = os.path.abspath(os.path.expanduser(cfg.OUTPUT_DIR))
            self.weight_dir = os.path.join(self.output_dir, cfg.MODEL.WEIGHT_PATH)
            
            self.dataset = ImageNet(cfg)
            self.model = Model(cfg, self.dataset.classnames)
        
        cfg = self.cfg
        round_id = 0
        if cfg.TEST.TEST_MODEL == "last_round":    
            round_id = cfg.FL.ROUNDS_TOTAL
        else:
            round_id = cfg.TEST.TEST_MODEL
        test_weights = os.path.join(self.weight_dir, f"model_round{round_id}_seed{self.seed}.pth.tar")
        model, _, _, _ = self.model.build_model(test_weights)
        test_dataloader = self.dataset.build_data_loader(is_train=False)
        lab2cname = self.dataset.lab2cname
        result_acc = self.model.test_model(test_dataloader, model, lab2cname)
        print("test done ...")
        
        
    
    def run_fl(self, eval_only=False): 
        if not eval_only:
            self.before_train()
            self.run_train()        
            self.after_train()
        
        self.test(eval_only)
        
    

if __name__ == "__main__":
    demo = FL()
    demo.run_fl(eval_only=False)
    