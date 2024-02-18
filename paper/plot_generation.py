#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%
import pandas as pd
import numpy as np

# Data
data = {
    'Learning Rate': [1e-4, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'Regularization': [0, 1, 10, 100, 1000, 10000],
    'Validation Perplexity': [16.8, 16.59, 12.85, 12.82, 13.26, 13.91]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define formatters
formatters = {
    'Learning Rate': lambda x: '$10^{{{}}}$'.format(int(np.log10(x))) if x != 0 else '0',
    'Regularization': lambda x: '$10^{{{}}}$'.format(int(np.log10(x))) if x != 0 else '0',
    'Validation Perplexity': '{:.2f}'.format
}

# Print DataFrame as LaTeX table with formatters
print(df.to_latex(index=False, formatters=formatters, escape=False))
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style defaults and set the default figure size
sns.set_theme(style='ticks', rc={'figure.figsize':(11.7,8.27)})

# Create the plot
plt.plot(df['Regularization'], df['Validation Perplexity'], marker='o', label='Validation Perplexity')

# Add labels and title
plt.xlabel('Regularization Constant (symlog scale)', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)

# Set x-axis to symlog scale
plt.xscale('symlog')

# Add gridlines
plt.grid(True)

# Add legend
plt.legend()

# Save the plot to a file with a transparent background
plt.savefig('figures/results_plot.png', bbox_inches='tight', dpi=300, transparent=True)
# %%