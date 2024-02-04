#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch
import data_utils
#%%
from torch.optim import AdamW, SGD
from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning as L
#from deepspeed.ops.adam import FusedAdam
from accelerate import Accelerator
import torch
import numpy as np

class LightningHier(L.LightningModule):
    """ PyTorch Lightning Model class for model training"""

    def __init__(
        self,
        tokenizer,
        model,
        task_stats,
        outputdir: str = "outputs",
        save_only_last_epoch: bool = False,
        learning_rate: float = 0.00001,
        accumulate_grad_batches: int = 1,
        weight_decay:  float = 0.0,
        reg_weight: float = 0.0,
        global_only: bool = False,
        *args, **kwargs
    ):
        """
        initiates a PyTorch Lightning Model
        """
        super().__init__()
        self.task_stats= task_stats
        self.tasks = list(task_stats.keys())
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.save_only_last_epoch = save_only_last_epoch
        self.learning_rate = learning_rate
        self.accumulate_grad_batches = accumulate_grad_batches
        self.weight_decay = weight_decay
        self.reg_weight = reg_weight
        self.global_only = global_only

    def forward(self, input_ids, attention_mask, labels=None, *args, **kwargs):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output
    
    def compute_l2_norms_of_adapters(self):
        l2_norms = {}
        for key in self.tasks:
            l2_norms[f"l2/{key}"] = 0
            l2_norms[f"l2_from_base/{key}"] = 0
        l2_norms["l2/base"] = 0.0

        for base_key, val in self.model.named_parameters():
            if "base_adapter" in base_key:
                l2_norms["l2/base"] += torch.norm(self.model.get_parameter(base_key))
                
                for adapter in self.tasks:
                    adapter_key = base_key.replace("base_adapter", adapter)
                    # compute l2 norm of adapter
                    l2_norms[f"l2/{adapter}"] += torch.norm(self.model.get_parameter(adapter_key))
                    # compute l2 norm between base adapter and adapter
                    diff = torch.norm(self.model.get_parameter(base_key) - self.model.get_parameter(adapter_key))
                    l2_norms[f"l2_from_base/{adapter}"] += diff
        return l2_norms
    
    #def on_validation_start(self, *args, **kwargs):
    #    l2_norms = self.compute_l2_norms_of_adapters()
    #    self.log_dict({f"l2_norm/{key}" : val for key, val in l2_norms.items()}, on_epoch=True, sync_dist=True)

    def step(self, batch, phase):
        batch_tasks = np.array(batch['task'])
        logs = {}
        loss_loglik_dict = {}
        for task in self.tasks:
            # task = self.tasks[0]
            current_task = (task == batch_tasks)
            if not current_task.any():
                continue
            # filter out current tasks if tensor or list
            batch_task = {key : val[current_task] 
                          for key, val in batch.items() 
                          if isinstance(val, torch.Tensor)}

            # Set adapter to relevant task except if global_only is True
            if not self.global_only:
                self.model.set_adapter(task)
            else:
                self.model.set_adapter("base_adapter")

            output = self(
                input_ids = batch_task['input_ids'], 
                attention_mask = batch_task['attention_mask'], 
                labels=batch_task['labels'])
            
            loss_loglik_dict[task] = output['loss']

        ### hierarchical loss
        self.model.set_adapter(self.tasks + ["base_adapter"]) # Set all adapters to active again
        
        l2_norms = self.compute_l2_norms_of_adapters()
        if phase=="train":
            self.log_dict(l2_norms, on_epoch=False, sync_dist=True, on_step=True)

        ## Compute total loss
        reg_loss = sum([val for key, val in l2_norms.items() if "l2_from_base" in key])
        
        loss_loglik = sum([val*self.task_stats[task]['train_len'] for task, val in loss_loglik_dict.items()])
        if self.reg_weight>0:
            loss = loss_loglik + reg_loss*self.reg_weight
        else:
            loss = loss_loglik
        ## LOGGING
        
        #
        for key, val in loss_loglik_dict.items():
            logs[f"{phase}_all/loglik/{key}"] = -val

        

        avg_loglik = -sum(loss_loglik_dict.values())/len(loss_loglik_dict)
        logs[f'{phase}/loglik'] = avg_loglik
        logs[f'{phase}/perplexity'] = torch.exp(-avg_loglik)
        
        logs[f'{phase}/loss'] = loss
        for key, val in logs.items():
            self.log(key, val, on_epoch=True, sync_dist=True if phase=="val" else False)
        
        """
        loss = loss_loglik
        self.model.set_adapter(self.tasks + ["base_adapter"])
        loss.backward()
        for key, par in self.model.named_parameters():
            if "base_adapter" in key:
                print("---")
                for adapter in  ['base_adapter'] + self.tasks:
                    adapter_key = key.replace("base_adapter", adapter)
                    adapter_par = self.model.get_parameter(adapter_key)
                    if adapter_par.grad is not None:
                        v = adapter_par.grad.norm().item()
                        print(adapter_key,"\t", v)
        self.model.zero_grad()        
        """

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, phase="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase="val")

    def configure_optimizers(self):
        """ configure optimizers """
        count_model_pars(self.model)
        return AdamW(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)


def count_model_pars(model):
    # count number of trainable parameters as share of total:
    num_pars, num_trainable = 0, 0
    for key, par in model.named_parameters():
        num_pars += par.numel()
        if par.requires_grad:
            num_trainable += par.numel()
    print(f"Model has {num_pars/1e6:.1f}m parameters of which {num_trainable/1e6:.1f}m are trainable ({num_trainable/num_pars*100:.2f}%)")
    return num_pars, num_trainable

def load_tokenizer(params):
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right' # needed for lama as they pad left
    return tokenizer

def load_model(params, tokenizer, task_stats, checkpoint_path=None):
    # Load model given parameters in params
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForCausalLM.from_pretrained(
        params['model_name'], 
        trust_remote_code=True, 
        load_in_8bit=params.get('load_in_8bit',False),
        load_in_4bit=params.get('load_in_4bit',False),
        device_map=device_map,
        )
    
    # Build adapters per task
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=params['lora_dim'], 
        lora_alpha=params['lora_alpha'], 
        lora_dropout=params['lora_dropout'],
        target_modules=params['lora_target_modules']
        )
    
    model = prepare_model_for_kbit_training(model)
    # Add adapters
    model.add_adapter(lora_config, adapter_name="base_adapter")
    
    for task in task_stats.keys():
        model.add_adapter(lora_config, adapter_name=task)
        count_model_pars(model)

    # Set parameters in adapters to be equivalent to base_adapter
    for base_key, val in model.named_parameters():
        if "base_adapter" in base_key:
            for adapter in task_stats.keys():
                adapter_key = base_key.replace("base_adapter", adapter)
                model.get_parameter(adapter_key).data = model.get_parameter(base_key).data.detach().clone()
                # Add some noise to the adapters
                model.get_parameter(adapter_key).data += torch.randn_like(model.get_parameter(adapter_key).data) * 0.001

    if checkpoint_path:
        print("loading from checkpoint..")
        pl_model = LightningHier.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, model=model, task_stats=task_stats, **params)
        del model
        torch.cuda.empty_cache()
    else:
        pl_model = LightningHier(tokenizer=tokenizer, model=model, task_stats=task_stats, **params)
    return pl_model
def main(overwrite_params={}):
    params = {
        'model_name': "facebook/opt-350m",
        'max_token_length': 256,
        'batch_size' : 64,
        'load_in_8bit' : False,
        'num_tasks' : 25,
        'reg_weight' : 10.1,
        "global_only" : False,
        # LORA PARAMETERS
        'lora_dim' : 16,
        'lora_alpha' : 16,
        'lora_dropout' : 0.0,
        'lora_target_modules' : None,
        # OPTIM PARAMETERS
        'learning_rate' : 0.0001,
        'weight_decay' : 0,
        'accumulate_grad_batches' : 1,
        'early_stopping_patience_epochs' : 8,
        'max_epochs' : 1000,
        'precision': 32,
        'log_every_n_steps' : 10,
        'val_check_interval' : 1.0, # 0.25 = 4 times per epoch
    }
    for key, val in overwrite_params.items():
        params[key] = val

    tokenizer = load_tokenizer(params)
    dataloaders, task_stats = data_utils.prepare_talkofnorway_dataloaders(tokenizer, **params)
    task_stats
    pl_model = load_model(params, task_stats=task_stats, tokenizer=tokenizer)
    #%%
    callbacks = [
        L.pytorch.callbacks.TQDMProgressBar(refresh_rate=5), 
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/loglik", mode="max", save_top_k=1),
        ]

    if params['early_stopping_patience_epochs'] > 0:
        early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
            monitor="val/loglik",
            min_delta=0.00,
            patience=params['early_stopping_patience_epochs'],
            verbose=True,
            mode="max",
        )
        callbacks.append(early_stop_callback)

    # prepare trainer
    print(params)
    trainer = L.Trainer(
        callbacks=callbacks,
        logger = L.pytorch.loggers.TensorBoardLogger("logs",name = f"1kepoch-reg:{params['reg_weight']}-lr:{params['learning_rate']}-global:{params['global_only']}-loradim:{params['lora_dim']}"),
        max_epochs=params['max_epochs'],
        #strategy="ddp",
        devices=1, 
        accelerator="gpu",
        accumulate_grad_batches=params['accumulate_grad_batches'],
        precision=params['precision'],
        val_check_interval=params['val_check_interval'],
        #limit_val_batches=1,
        log_every_n_steps=params['log_every_n_steps']
    )
    #%%
    trainer.fit(pl_model,
                train_dataloaders=dataloaders['train'], 
                val_dataloaders=dataloaders['valid'])
    
if __name__ == "__main__":
    main(overwrite_params={"reg_weight" : 0, "global_only" : True, "learning_rate" : 0.00001,'lora_dim' : 2,'lora_alpha' : 2})

    l1 = [0.1,1,5,10]
    l2 = [1000,100,10]
    for regloss in l1:
        main(overwrite_params={"reg_weight" : regloss, "learning_rate" : regloss*0.00001,'lora_dim' : 2,'lora_alpha' : 2})
    #main()
#%% RUN FOR INTERACTIVE
"""
batch = next(iter(dataloaders['train']))
batch = {key : val.to(pl_model.model.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
self = pl_model
phase="train"
"""

