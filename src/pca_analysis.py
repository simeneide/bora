#%% RUN 
from train import *
"""
overwrite_params = {}
"""
params = {
    'model_name': "facebook/opt-350m",
    'max_token_length': 256,
    'batch_size' : 64,
    'load_in_8bit' : False,
    'num_tasks' : 25,
    'reg_weight' : 1.0,
    "optim_type" : "regularized", # global_only, joined_grad , regularized
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

tokenizer = load_tokenizer(params)
dataloaders, task_stats = data_utils.prepare_talkofnorway_dataloaders(tokenizer, **params)

pl_model = load_model(params, task_stats=task_stats, tokenizer=tokenizer, checkpoint_path="/home/simen.eide@schibsted.com/hier-llm/src/logs_regularized/1kepoch-reg:10-lr:0.0001-global:False/version_1/checkpoints/epoch=281-step=22560.ckpt")

model = pl_model.model
#%% concat all adapter weights from different tasks
parameter_store = {key : [] for key in task_stats.keys()}
parameter_away_from_base_store = {key : [] for key in task_stats.keys()}
for base_key, val in model.named_parameters():
    if "base_adapter" in base_key:
        for adapter in task_stats.keys():
            adapter_key = base_key.replace("base_adapter", adapter)
            par = model.get_parameter(adapter_key).data
            # Add to list
            parameter_store[adapter].append( par.flatten() )
            # Compare to base and add to list
            base_par = model.get_parameter(base_key).data
            parameter_away_from_base_store[adapter].append( (par - base_par).flatten() )

adapter_weights = torch.stack([torch.concat(parameter_store[task]) for task in task_stats.keys()])

adapter_weights_away_from_base = torch.stack([torch.concat(parameter_away_from_base_store[task]) for task in task_stats.keys()])

# print(adapter_weights.shape)
# (25, 1572864)

#%% Compare the norm of the weights compared to training data lengths
import numpy as np
weight_norms = adapter_weights_away_from_base.norm(dim=1).cpu().numpy()
data_lengths = np.array([task_stats[task]["train_len"] for task in task_stats.keys()])
# Visualize in a plot
import matplotlib.pyplot as plt
import seaborn as sns

# Set a theme


# Create a new figure and set its size

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
# Create a new figure and set its size
fig, ax = plt.subplots(figsize=(10, 8))
# Create a scatter plot
scatter = ax.scatter(data_lengths, weight_norms, alpha=0.5)

# Set the title and labels
ax.set_title('Distance of Adapter Weights from Prior vs Training Data Lengths', fontsize=14)
ax.set_xlabel('Training Data Lengths', fontsize=12)
ax.set_ylabel('Distance of Adapter Weights from Prior (l2-norm)', fontsize=12)

# Increase the size of the ticks labels
ax.tick_params(axis='both', which='major', labelsize=10)

# Add a grid
plt.grid(True)

# Remove the top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save the plot with a transparent background
plt.savefig('weights_vs_data_lengths.png', bbox_inches='tight', dpi=300, transparent=True)

# Show the plot
plt.show()
#%% PCA
#############
adapter_weights_np = adapter_weights.cpu().numpy()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(adapter_weights_np)
transformed_weights = pca.transform(adapter_weights_np)
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the demographics data
dem = pd.read_csv("demographics.csv")

# Create a color map
cmap = sns.color_palette('hsv', dem['hjemfylke'].nunique())

# Create a dictionary to map each county to a color
color_dict = dict(zip(dem['hjemfylke'].unique(), cmap))

# Create a scatter plot
plt.figure(figsize=(10, 10))

for i, task in enumerate(task_stats.keys()):
    # Get the county for the current task
    county = dem.loc[dem['navn'] == task, 'hjemfylke'].values[0]
    
    # Get the color for the county
    color = color_dict[county]
    
    plt.scatter(transformed_weights[i, 0], transformed_weights[i, 1], color=color)
    plt.text(transformed_weights[i, 0], transformed_weights[i, 1], task)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Adapter Weights')

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=8) for v in color_dict.values()]
plt.legend(handles, color_dict.keys(), title='Home County', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig('pca_weights.png')

plt.show()
#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load the demographics data
dem = pd.read_csv("demographics.csv")

# Create a color map
cmap = sns.color_palette('hsv', dem['hjemfylke'].nunique())

# Create a dictionary to map each county to a color
color_dict = dict(zip(dem['hjemfylke'].unique(), cmap))

# Create a scatter plot
plt.figure(figsize=(10, 10))

# Group the speakers by county and calculate the average vector for each group
grouped_weights = {}
for i, task in enumerate(task_stats.keys()):
    # Get the county for the current task
    county = dem.loc[dem['navn'] == task, 'hjemfylke'].values[0]
    
    # Add the weights to the group
    if county not in grouped_weights:
        grouped_weights[county] = []
    grouped_weights[county].append(transformed_weights[i])

# Calculate the average vector for each group and plot
for county, weights in grouped_weights.items():
    avg_weight = np.mean(weights, axis=0)
    plt.scatter(avg_weight[0], avg_weight[1], color=color_dict[county], edgecolors='black', linewidths=2, s=100)
    plt.text(avg_weight[0], avg_weight[1], county)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Adapter Weights')
# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=8) for v in color_dict.values()]
plt.legend(handles, color_dict.keys(), title='Home County', bbox_to_anchor=(1.05, 1), loc='upper left')
# Save the plot
plt.savefig('pca_weights.png')
plt.show()
# %%
