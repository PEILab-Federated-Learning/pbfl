import os
from configs.imagenet_vitb16 import _C
from dataloader.imagenet import ImageNet
from model.model import Model
from dataloader.utils.tools import mkdir_if_missing

class FL:
    clients_dataloader = []
    
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
            
        self.model = Model(cfg, self.dataset)
        print("model (clip) done ...")

    def run_train(self):
        print("run_train starting ...")
        cfg = self.cfg
        for round_id in range(cfg.FL.ROUNDS_TOTAL):
            print('Starting fl round', '['+str(round_id+1)+'/'+str(cfg.FL.ROUNDS_TOTAL)+']', '...')
            self.before_round(round_id)
            self.run_round(round_id)
            self.after_round(round_id)
            
            return

    def after_train(self):
        pass


    def before_round(self, round_id):
        pass
        
        

    def after_round(self, round_id):
        pass

    def run_round(self, round_id):
        cfg = self.cfg
        for client_id in range(cfg.FL.CLIENTS_TOTAL):
            print('Starting fl client', '['+str(client_id+1)+'/'+str(cfg.FL.CLIENTS_TOTAL)+']', '...')
            
            client_dataloader = self.clients_dataloader[client_id]
            if round_id == 0:
                self.model.build_model()
            else:
                weight_dir = os.path.join(self.output_dir, cfg.MODEL.WEIGHT_PATH)
                mkdir_if_missing(weight_dir)
                weight_file = os.path.join(weight_dir, f"model_round{round_id-1}_seed{self.seed}.pth.tar")
                self.model.build_model(weight_file)

    def run_fl(self):     
        self.before_train()
        self.run_train()
        
        return
        
        self.after_train()
        
    

if __name__ == "__main__":
    demo = FL()
    demo.run_fl()