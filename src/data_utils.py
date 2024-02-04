#%%
from datasets import load_dataset, DatasetDict
import datasets
import transformers
from torch.utils.data import DataLoader
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
import pandas as pd

def tokenize_features(sample, tokenizer, features=None, max_token_len=None):
    if features is None:
        features = sample.keys()

    batch = {}
    for key, val in sample.items():
        batch[key] = f" [{key}] {val}"
        if key in features:
            batch[f"{key}.input_ids"] = tokenizer(
                batch[key],
                return_attention_mask=False,
                add_special_tokens=False,
                max_length=max_token_len,
                truncation=True
            )['input_ids']
    return batch

def concat_features(sample, out_config, tokenizer, max_token_len):
    batch = {}
    #batch = {key:val for key, val in sample.items() if "input_ids" not in key}

    for config in out_config:
        features = config['input_features'] + config['trainable_features']

        tot_tokens = sum([len(sample[f"{f}.input_ids"]) for f in features])
        excessive_tokens = tot_tokens - max_token_len +1 # +1 for eos token
        if excessive_tokens>0:
            sample[f"input_ids.{config['feature_to_trunc']}"] = sample[f"{config['feature_to_trunc']}.input_ids"][:-excessive_tokens]
        if config['token_prepend']!="":
            token_prepend = f"{config['token_prepend']}."
        else:
            token_prepend = ""
        batch[f"{token_prepend}input_ids"] = []
        batch[f"{token_prepend}labels"] = []
        for f in config['input_features']:
            input_col = f"{token_prepend}input_ids"
            label_col = f"{token_prepend}labels"
            batch[input_col] += sample[f"{f}.input_ids"]
            batch[label_col] += [-100]*len(sample[f"{f}.input_ids"])

        for tf in config['trainable_features']:
            batch[input_col] += sample[f"{tf}.input_ids"]
            batch[label_col] += sample[f"{tf}.input_ids"]
        
        # Append eos token
        batch[input_col] += [tokenizer.eos_token_id]
        batch[label_col] += [tokenizer.eos_token_id]
        
        # Truncate at end
        batch[input_col] = batch[input_col][:max_token_len]
        batch[label_col] = batch[label_col][:max_token_len]

    return batch

def tokenize_and_concat(sample, tokenizer, out_config, max_token_len=512):
    batch = tokenize_features(sample,tokenizer, max_token_len=max_token_len-10) # -10 to leave room for other features
    batch = concat_features(batch, out_config, tokenizer=tokenizer, max_token_len=max_token_len)
    return batch

#%%
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

def add_task_feature(datasetDict, task_name):
    datasetDict = datasetDict.map(lambda sample: {'task': task_name}, num_proc=None)
    return datasetDict

def load_nbnn_dataset(tokenizer, max_token_len, train_split="train[:1000]", task_name="nbnn"):
    dataset_nbnn = DatasetDict({
        'train' : load_dataset("NbAiLab/nbnn_translation", split=train_split),
        'valid' : load_dataset("NbAiLab/nbnn_translation", split="dev"),
        'test' : load_dataset("NbAiLab/nbnn_translation", split="test")
    })

    out_config = [
    {
        'token_prepend' : 'main',
        'input_features' : ['task','nbo'],
        'trainable_features' : ['nno'],
        'feature_to_trunc' : 'nbo'
    }
    ]

    dataset_nbnn = add_task_feature(dataset_nbnn, task_name)

    dataset_nbnn = dataset_nbnn.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=6) # 
    return dataset_nbnn

def load_parliament_dataset(tokenizer, max_token_len, train_split="train[:1000]", task_name="parliament"):
    dataset_raw = DatasetDict({
        'train' : load_dataset("NbAiLab/norwegian_parliament", split=train_split),
        'valid' : load_dataset("NbAiLab/norwegian_parliament", split="validation"),
        'test' : load_dataset("NbAiLab/norwegian_parliament", split="test")
    })
    out_config = [
    {
        'token_prepend' : 'main',
        'input_features' : ['task','label'],
        'trainable_features' : ['text'],
        'feature_to_trunc' : 'text'
    },
    ]

    dataset_raw = add_task_feature(dataset_raw, task_name)

    dataset_raw = dataset_raw.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=6) # 
    return dataset_raw



def load_noralpaca(tokenizer, max_token_len):
    dataset_raw = load_dataset("NbAiLab/norwegian-alpaca")['train']

    # Split into train/val/test
    split_1 = dataset_raw.train_test_split(test_size=0.1, seed=42)
    dataset_valid = split_1['test']
    split_2 = split_1['train'].train_test_split(test_size=0.1, seed=42)
    dataset_test = split_2['test']
    dataset_train = split_2['train']

    dataset_noralpaca = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})
    dataset_noralpaca = add_task_feature(dataset_noralpaca, "noralpaca")
    
    out_config = [
    {
        'token_prepend' : 'main',
        'input_features' : ['task','instruction','input'],
        'trainable_features' : ['output'],
        'feature_to_trunc' : 'input'
    },
    ]
    dataset_noralpaca = dataset_noralpaca.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=None)
    return dataset_noralpaca


def load_talkofnorway_dataset(tokenizer, max_token_len, min_token_length=10, num_tasks=3):
    """
    num_tasks = 3 # cut the number of speakers
    max_token_len=512 
    min_token_length=50
    """
    min_examples, max_examples = 200, 1000
    raw = (pd.read_csv("data/ton.csv")
        # select columns full name, party affiliation, time and text
        .pipe(lambda df: df[['rep_name', 'text']]) # 'time','party_name', 
        .pipe(lambda df: df.dropna())
        # only keep speakers that have between 100 and 500 examples
        .pipe(lambda df: df.groupby('rep_name').filter(lambda x: min_examples < len(x) < max_examples))
        # roughly remove text that is shorter than min_token_length tokens
        .pipe(lambda df: df[df['text'].str.len() > 4*min_token_length])
        #.pipe(lambda df: df[df['text'].str.len() < 4*max_token_len])
        # create feature "task" to be lowercase and only letters, not even punctuation
        .pipe(lambda df: df.assign(task=df['rep_name'].str.replace('.', '').str.replace('[^a-z]', '').str.lower()))
        
        # Reset index
        .pipe(lambda df: df.reset_index(drop=True))
        )
    """ Raw looks like
    time	task	party_name	text
    0	1998-10-20T00:00:00+02:00	Sonja Irene Sjøli	Høyre	Det er en bred forståelse blant fagfolk og pol...
    1	1998-10-20T00:00:00+02:00	Dagfinn Høybråten	Kristelig Folkeparti	Et av hovedmålene for helsetjenesten i Norge e...
    2	1998-10-20T00:00:00+02:00	Sonja Irene Sjøli	Høyre	Jeg takker helseministeren for et fyldig og gr...
    3	1998-10-20T00:00:00+02:00	Sonja Irene Sjøli	Høyre	
    """

    # Group by task and create train and val datasets for each
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    task_stats = {}

    for task, df in raw.groupby('task'):
        # 50% train, 25% val, 25% test
        train_df = df.sample(frac=0.5, random_state=42)
        temp_df = df.drop(train_df.index)
        val_df = temp_df.sample(frac=0.5, random_state=42)
        test_df = temp_df.drop(val_df.index)
        train_dfs[task], val_dfs[task], test_dfs[task] = train_df, val_df, test_df

        task_stats[task] = {
            'train_len' : len(train_df),
            'val_len' : len(val_df),
            'test_len' : len(test_df),
        }
        if len(train_dfs) == num_tasks:
            break
    # Concatenate all train and val datasets
    train_df = pd.concat(train_dfs.values()).reset_index(drop=True)
    val_df = pd.concat(val_dfs.values()).reset_index(drop=True)
    test_df = pd.concat(test_dfs.values()).reset_index(drop=True)

    # Convert into datasets format
    ds = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df),
        'valid': datasets.Dataset.from_pandas(val_df),
        'test': datasets.Dataset.from_pandas(test_df)
    })

    out_config = [
    {
        'token_prepend' : '',
        'input_features' : ['task'],
        'trainable_features' : ['text'],
        'feature_to_trunc' : 'text'
    },
    ]
    ds = ds.map(lambda sample: tokenize_and_concat(sample, tokenizer,out_config,max_token_len=max_token_len), num_proc=None)

    return ds, task_stats

def prepare_talkofnorway_dataloaders(tokenizer, batch_size=4, max_token_len=128, num_tasks=3, *args, **kwargs):
    """
    batch_size=4
    max_token_len=128
    num_tasks=3
    """
    ds, task_stats = load_talkofnorway_dataset(tokenizer, max_token_len=max_token_len, num_tasks=num_tasks)
    ds = ds.shuffle(seed=42)
    data_collate = DataCollatorWithPaddingAndLabels(tokenizer=tokenizer, max_length=max_token_len)
    dataloaders = {}
    for phase in ["train","valid","test"]:
        dataloaders[phase] = DataLoader(ds[phase], batch_size=batch_size, collate_fn=data_collate)
    return dataloaders, task_stats

def prepare_dataloaders(tokenizer, batch_size=4, max_token_len=128, *args, **kwargs):
    """
    batch_size = 4
    max_token_len = 128
    """
    dataset_dict = {
    'nbnn10k' : load_nbnn_dataset(tokenizer, max_token_len=max_token_len, train_split="train[:10000]", task_name="nbnn10k"),
    'nbnn500' : load_nbnn_dataset(tokenizer, max_token_len=max_token_len, train_split="train[10000:10500]", task_name="nbnn500"),
    'parliament' : load_parliament_dataset(tokenizer, max_token_len=max_token_len, train_split="train", task_name="parliament"),
    }
    data_collate = DataCollatorWithPaddingAndLabels(tokenizer=tokenizer, max_length=max_token_len)
    ds = {}
    dataloaders = {}
    for phase in ["train","valid","test"]:
        ds[phase] = datasets.concatenate_datasets([d[phase] for d in dataset_dict.values()]).shuffle(seed=42)
        dataloaders[phase] = DataLoader(ds[phase], batch_size=batch_size, collate_fn=data_collate)
    return dataloaders
#%%
if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataloaders, task_stats = prepare_talkofnorway_dataloaders(tokenizer, batch_size=4, max_token_len=512, num_tasks=3)
    #dataloaders = prepare_dataloaders(tokenizer, batch_size=4, max_token_len=128)
    
    for batch in dataloaders['train']:
        print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['labels'].shape)
        assert(batch['input_ids'].shape == batch['attention_mask'].shape == batch['labels'].shape)
        for l in tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False):
            print(l)
        break
