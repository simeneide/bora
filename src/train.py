#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch
import data_utils

params = {
    'model_name': "facebook/opt-350m",
    'max_token_length': 256,
    'batch_size' : 4,
    'load_in_8bit' : True,
    'tasks' : ["nbnn10k","parliament", "nbnn500"],

    # LORA PARAMETERS
    'lora_dim' : 16,
    'lora_alpha' : 16,
    'lora_dropout' : 0.0,
    'lora_target_modules' : None,
    # OPTIM PARAMETERS
    'learning_rate' : 0.0001,
    'weight_decay' : 0,
    'batch_size' : 32,
    'accumulate_grad_batches' : 1,
    'early_stopping_patience_epochs' : 5,
    'max_epochs' : 100,
    'precision': 32,
    'log_every_n_steps' : 10,
    'val_check_interval' : 0.25, # 0.25 = 4 times per epoch
    
}
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
        tasks,
        outputdir: str = "outputs",
        save_only_last_epoch: bool = False,
        learning_rate: float = 0.00001,
        accumulate_grad_batches: int = 1,
        weight_decay:  float = 0.0,
        *args, **kwargs
    ):
        """
        initiates a PyTorch Lightning Model
        """
        super().__init__()
        self.tasks = tasks
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch
        self.learning_rate = learning_rate
        self.accumulate_grad_batches = accumulate_grad_batches
        self.weight_decay = weight_decay

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
        loss_loglik = 0
        for task in self.tasks:
            # task = self.tasks[0]
            current_task = (task == batch_tasks)
            if not current_task.any():
                continue
            # filter out current tasks if tensor or list
            batch_task = {key : val[current_task] 
                          for key, val in batch.items() 
                          if isinstance(val, torch.Tensor)}
            #batch['main.input_ids'][current_task].shape

            self.model.set_adapter(task)
            output = self(
                input_ids = batch_task['main.input_ids'], 
                attention_mask = batch_task['main.attention_mask'], 
                labels=batch_task['main.labels'])

            logs[f"{phase}/loglik/{task}"] = -output['loss']
            loss_loglik += output['loss']
        
        #
        # hierarchical loss
        with torch.no_grad():
            self.model.set_adapter(self.tasks + ["base_adapter"]) # Set all adapters to active again
            
            l2_norms = self.compute_l2_norms_of_adapters()
            if phase=="train":
                self.log_dict({f"l2_norm/{key}" : val for key, val in l2_norms.items()}, on_epoch=False, sync_dist=True, on_step=True)

            reg_loss = sum([val for key, val in l2_norms.items() if "base" not in key])
            
            logs[f'{phase}/regloss'] = reg_loss
        DO_HIER_LOSS = False
        if DO_HIER_LOSS:
            loss = loss_loglik + reg_loss
        else:
            loss = loss_loglik
            

        ## LOGGING
        logs[f'{phase}/loglik'] = -loss_loglik
        
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
                    else:
                        v=None
                    print(adapter_key,"\t", v)


        self.model.zero_grad()
        """

        return loss
    def on_before_backward(self, loss):
        self.model.set_adapter(self.tasks + ["base_adapter"])

    def training_step(self, batch, batch_idx):
        return self.step(batch, phase="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase="val")

    def configure_optimizers(self):
        """ configure optimizers """
        self.model.set_adapter(self.tasks) #  + ["base_adapter"]
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

def load_model(params, checkpoint_path=None):
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
    
    for task in params['tasks']:
        model.add_adapter(lora_config, adapter_name=task)
        count_model_pars(model)

    # Set parameters in adapters to be equivalent to base_adapter
    for base_key, val in model.named_parameters():
        if "base_adapter" in base_key:
            for adapter in params['tasks']:
                adapter_key = base_key.replace("base_adapter", adapter)
                model.get_parameter(adapter_key).data = model.get_parameter(base_key).data.detach().clone()
                # Add some noise to the adapters
                model.get_parameter(adapter_key).data += torch.randn_like(model.get_parameter(adapter_key).data) * 0.001

    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right' # needed for lama as they pad left

    if checkpoint_path:
        print("loading from checkpoint..")
        pl_model = LightningHier.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, model=model, **params)
        del model
        torch.cuda.empty_cache()
    else:
        pl_model = LightningHier(tokenizer=tokenizer, model=model, **params)
    return pl_model

pl_model = load_model(params)
dataloaders = data_utils.prepare_dataloaders(tokenizer=pl_model.tokenizer, **params)


#%%
callbacks = [
    L.pytorch.callbacks.TQDMProgressBar(refresh_rate=5), 
    L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss"),
    ]

if params['early_stopping_patience_epochs'] > 0:
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=params['early_stopping_patience_epochs'],
        verbose=True,
        mode="min",
    )
    callbacks.append(early_stop_callback)

# prepare trainer
print(params)
trainer = L.Trainer(
    callbacks=callbacks,
    logger = L.pytorch.loggers.TensorBoardLogger("logs"),
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
trainer.fit(pl_model, 
            train_dataloaders=dataloaders['train'], 
            val_dataloaders=dataloaders['valid'])
#%% RUN FOR INTERACTIVE
batch = next(iter(dataloaders['train']))
batch = {key : val.to(pl_model.model.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
self = pl_model
phase="train"
# %%
