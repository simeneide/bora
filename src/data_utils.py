#%%
from datasets import load_dataset
import transformers
from torch.utils.data import DataLoader

def tokenize_features(sample, tokenizer, features=None, max_token_len=None):
    if features is None:
        features = sample.keys()

    batch = {}
    for key, val in sample.items():
        batch[key] = f" [{key}] {val}"
        if key in features:
            batch[f"input_ids.{key}"] = tokenizer(
                batch[key],
                return_attention_mask=False,
                add_special_tokens=False,
                max_length=max_token_len,
                truncation=True
            )['input_ids']
    return batch

#dataset_nbnn = dataset_nbnn.map(lambda sample: tokenize_features(sample, tokenizer), num_proc=6)

def concat_features(sample, out_config, tokenizer, max_token_len):
    batch = {}
    #batch = {key:val for key, val in sample.items() if "input_ids" not in key}

    for config in out_config:
        features = config['input_features'] + config['trainable_features']

        tot_tokens = sum([len(sample[f"input_ids.{f}"]) for f in features])
        excessive_tokens = tot_tokens - max_token_len +1 # +1 for eos token
        if excessive_tokens>0:
            sample[f"input_ids.{config['feature_to_trunc']}"] = sample[f"input_ids.{config['feature_to_trunc']}"][:-excessive_tokens]

        task_name = config['task_name']
        batch[f"input_ids.{task_name}"] = []
        batch[f"labels.{task_name}"] = []
        for f in config['input_features']:
            input_col = f"input_ids.{config['task_name']}"
            label_col = f"labels.{config['task_name']}"
            batch[input_col] += sample[f"input_ids.{f}"]
            batch[label_col] += [-100]*len(batch[input_col])

        for tf in config['trainable_features']:
            batch[input_col] += sample[f"input_ids.{tf}"]
            batch[label_col] += sample[f"input_ids.{tf}"]
        
        # Append eos token
        batch[input_col] += [tokenizer.eos_token_id]
        batch[label_col] += [tokenizer.eos_token_id]

    return batch

def tokenize_and_concat(sample, tokenizer, out_config, max_token_len=512):
    batch = tokenize_features(sample,tokenizer, max_token_len=max_token_len-10) # -10 to leave room for other features
    batch = concat_features(batch, out_config, tokenizer=tokenizer, max_token_len=max_token_len)
    return batch

#%%
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
#from transformers import DPODataCollatorWithPadding
@dataclass
class DataCollatorWithPaddingAndLabels:
    """
    Modified from the original DataCollatorWithPadding to handle labels and to send all other features to the batch without any processing.
    Data collator that will dynamically pad the inputs received. 

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer : PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, samples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # convert into a dict with lists
        features_list = {key: [] for key in samples[0].keys()}
        for sample in samples:
            for key, val in sample.items():
                features_list[key].append(val)
        
        batch = {}
        for key, val in features_list.items():
            if "input_ids" in key:
                padded = self.tokenizer.pad(
                    {'input_ids' : val},
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                    return_attention_mask=True
                )
                batch[key] = padded['input_ids']
                batch[key.replace("input_ids","attention_mask")] = padded['attention_mask']
            elif "labels" in key:
                batch[key] = self.tokenizer.pad(
                    {'input_ids' : val},
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                    return_attention_mask=False
                )['input_ids']
            else:
                batch[key] = val
        return batch

def load_nbnn_dataset(tokenizer, max_token_len):
    dataset_nbnn = load_dataset("NbAiLab/nbnn_translation")
    out_config = [
    {
        'task_name' : 'main',
        'input_features' : ['nbo'],
        'trainable_features' : ['nno'],
        'feature_to_trunc' : 'nbo'
    },
]
    # rename dev to valid
    dataset_nbnn['valid'] = dataset_nbnn['dev']
    del dataset_nbnn['dev']

    dataset_nbnn = dataset_nbnn.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=6) # 
    return dataset_nbnn

def load_noralpaca(tokenizer, max_token_len):
    dataset_noralpaca = load_dataset("NbAiLab/norwegian-alpaca")
    out_config = [
    {
        'task_name' : 'main',
        'input_features' : ['instruction','input'],
        'trainable_features' : ['output'],
        'feature_to_trunc' : 'input'
    },
    ]
    dataset_noralpaca = dataset_noralpaca.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=None)
    return dataset_noralpaca
#%%
if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset_nbnn = load_nbnn_dataset(tokenizer, max_token_len=128)
    dataset_noralpaca = load_noralpaca(tokenizer, max_token_len=128)
    import datasets
    ds = datasets.interleave_datasets([dataset_nbnn['train'], dataset_noralpaca['train']])
    
    data_collate = DataCollatorWithPaddingAndLabels(tokenizer=tokenizer, max_length=128)
    dataloader = DataLoader(ds, batch_size=4, collate_fn=data_collate)
    
    dataloaders = {key: DataLoader(val, batch_size=4, collate_fn=data_collate) for key, val in dataset.items()}

    for batch in dataloader:
        batch['input_ids.main'].shape
        batch['labels.main']
