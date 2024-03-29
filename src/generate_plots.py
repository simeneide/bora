#%% data
import numpy as np
import pandas as pd
from tbparse import SummaryReader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def build_dataset_of_dataset_sizes():
    # Gathered from the dataloader but pasted here for convenience
    task_stats = {'abid q raja': {'train_len': 169, 'val_len': 84, 'test_len': 85},
    'akhtar chaudhry': {'train_len': 110, 'val_len': 54, 'test_len': 55},
    'aksel hagen': {'train_len': 136, 'val_len': 68, 'test_len': 67},
    'alf egil holmelid': {'train_len': 128, 'val_len': 64, 'test_len': 63},
    'anders b werp': {'train_len': 145, 'val_len': 72, 'test_len': 73},
    'andré n skjelstad': {'train_len': 172, 'val_len': 86, 'test_len': 86},
    'andré oktay dahl': {'train_len': 244, 'val_len': 122, 'test_len': 122},
    'anette trettebergstuen': {'train_len': 120, 'val_len': 60, 'test_len': 59},
    'anita apelthun sæle': {'train_len': 128, 'val_len': 64, 'test_len': 63},
    'anne tingelstad wøien': {'train_len': 181, 'val_len': 90, 'test_len': 91},
    'anniken huitfeldt': {'train_len': 384, 'val_len': 192, 'test_len': 192},
    'ansgar gabrielsen': {'train_len': 404, 'val_len': 202, 'test_len': 203},
    'arild grande': {'train_len': 160, 'val_len': 80, 'test_len': 81},
    'arne lyngstad': {'train_len': 125, 'val_len': 62, 'test_len': 63},
    'arve kambe': {'train_len': 158, 'val_len': 78, 'test_len': 79},
    'asmund kristoffersen': {'train_len': 152, 'val_len': 76, 'test_len': 77},
    'audun lysbakken': {'train_len': 452, 'val_len': 226, 'test_len': 227},
    'bendiks h arnesen': {'train_len': 238, 'val_len': 118, 'test_len': 119},
    'bente thorsen': {'train_len': 147, 'val_len': 74, 'test_len': 73},
    'bjørg tørresdal': {'train_len': 214, 'val_len': 106, 'test_len': 107},
    'bjørn hernæs': {'train_len': 146, 'val_len': 72, 'test_len': 73},
    'bjørn jacobsen': {'train_len': 234, 'val_len': 118, 'test_len': 117},
    'bjørn lødemel': {'train_len': 129, 'val_len': 64, 'test_len': 65},
    'bjørn tore godal': {'train_len': 114, 'val_len': 58, 'test_len': 57},
    'borghild tenden': {'train_len': 477, 'val_len': 238, 'test_len': 239}}
    dataset_sizes = pd.DataFrame([{'task' : key, 'dataset_size' : val['train_len']} for key, val in task_stats.items()])
    return dataset_sizes
#%%

def get_perplex_per_task_tensorboard(experiment_id):
    log_dir = f"logs/{experiment_id}"
    reader = SummaryReader(log_dir)
    df = reader.scalars

    # Check if tag column contains "val_all/perplexity"
    val_all = df[df["tag"].str.contains("val_all")]
    val_all['task'] = val_all['tag'].str.extract(r'val_all/loglik/(.*)')

    # Group by task and get the maximum value of the value column
    val_all_max = val_all.groupby('task')['value'].max()
    # Combine val_all_max with experiment ID into a dataframe
    df = pd.DataFrame(val_all_max).reset_index()
    df['val_perplexity'] = np.exp(-df['value'])
    df['experiment_id'] = experiment_id
    return df


def get_perplexity_for_all_tasks(main_results):
    loglik_pds = []
    for experiment_id in main_results["experiment_id"]:
        loglik_pds.append(get_perplex_per_task_tensorboard(experiment_id))

    # Concat all df_task_perplexity into one dataframe
    df_task_perplexity = pd.concat(loglik_pds)[['task','val_perplexity','experiment_id']]
    # pivot the dataframe to experiments and tasks
    task_pivot = df_task_perplexity.pivot(index="experiment_id", columns="task", values="val_perplexity")
    df = task_pivot.merge(main_results, on="experiment_id")
    return df_task_perplexity, df

#%% ### ###
### Precompute values
### ### ###
main_results = pd.read_csv("main_results.csv")
df_task_perplexity, df = get_perplexity_for_all_tasks(main_results)
dataset_sizes = build_dataset_of_dataset_sizes()
tasks = df_task_perplexity['task'].unique()
sns.set_theme(style='ticks', rc={'figure.figsize':(11.7,8.27)})

#%% ### ###
### Generate latex table with main result
### ### ###

df_main_results = pd.DataFrame(main_results)

# Define formatters
formatters = {
    'Learning Rate': lambda x: '$10^{{{}}}$'.format(int(np.log10(x))) if x != 0 else '0',
    'Regularization': lambda x: '$10^{{{}}}$'.format(int(np.log10(x))) if x != 0 else '0',
    'Validation Perplexity': '{:.2f}'.format
}

# Print DataFrame as LaTeX table with formatters
print(df_main_results.to_latex(index=False, formatters=formatters, escape=False))

#%% ### ###
###  Plot training data length vs perplexity for the optimal model
### ### ###
# Hypothesis: Will tasks with more data generally have lower perplexity than tasks with less data?
experiment_id = "1kepoch-reg:100-lr:0.001-global:False/version_0"
task_size_vs_perplexity = (
    df_task_perplexity[df_task_perplexity['experiment_id'] == experiment_id]
    .merge(dataset_sizes, left_on='task', right_on='task')
)

task_size_vs_perplexity.plot(x='dataset_size', y='val_perplexity', kind='scatter')
plt.xlabel('Training data length')
plt.ylabel('Perplexity')
plt.title('Training data length vs Perplexity')
plt.grid(True)
plt.savefig('figures/trainingsize_vs_perplexity.png', bbox_inches='tight', dpi=300, transparent=True)
# result: no big change here

#%% ### ###  
# Plot relative improvement of using hierarchical model vs separate models. compare with data size
### ### ###

best_experiment_id = "1kepoch-reg:100-lr:0.001-global:False/version_0"
best_model = df_task_perplexity[df_task_perplexity['experiment_id'] == best_experiment_id].sort_values("task")
separate_model = df_task_perplexity[df_task_perplexity['experiment_id'] == "1kepoch-reg:0-lr:0.0001-global:False/version_0"].sort_values("task")

one_model = df_task_perplexity[df_task_perplexity['experiment_id'] == "1kepoch-reg:10000-lr:0.1-global:False-loradim:16/version_0"].sort_values("task")

improvement_separate = 1-best_model['val_perplexity']/separate_model['val_perplexity']
improvement_one = 1-best_model['val_perplexity']/one_model['val_perplexity']

task_size_vs_improvement = pd.DataFrame({
    'task': best_model['task'], 
    'improvement_separate': improvement_separate, 
    'improvement_one': improvement_one
    }).merge(dataset_sizes, left_on='task', right_on='task')

task_size_vs_improvement.plot(x='dataset_size', y='improvement_separate', kind='scatter')
plt.xlabel('Training data length')
plt.ylabel('Relative improvement from separate models')
#plt.title('Training data length vs Relative improvement from separate models')
plt.savefig('figures/relative_improvement_separate_models.png', bbox_inches='tight', dpi=300, transparent=True)

task_size_vs_improvement.plot(x='dataset_size', y='improvement_one', kind='scatter')
plt.xlabel('Training data length')
plt.ylabel('Relative improvement from one model')
#plt.title('Training data length vs Relative improvement from one model')
plt.savefig('figures/relative_improvement_one_model.png', bbox_inches='tight', dpi=300, transparent=True)
#%% ### ###
# PLOT REGULARIZATION VS PERPLEXITY FOR ALL TASKS
### ### ###

for task in tasks:
    plt.plot(df["regularization"], df[task], label=task, alpha=0.5, color="grey")

# Create the plot
plt.plot(df['regularization'], df['validation perplexity'], marker='o', label='Validation Perplexity', color="black", linewidth=2.5)
# Add labels and title
plt.xlabel('Regularization Constant (symlog scale)', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
# add horizontal line

# Set x-axis to symlog scale
plt.xscale('symlog')

# Add gridlines
plt.grid(True)
plt.savefig('figures/results_plot.png', bbox_inches='tight', dpi=300, transparent=True)
# Add legend
#%%