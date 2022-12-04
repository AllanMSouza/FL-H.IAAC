from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(df, base_dir, file_name, x_column, y_column, title, hue=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    sns.set(style='whitegrid')
    figure = sns.barplot(x=x_column, y=y_column, data=df).set_title(title)
    figure = figure.get_figure()
    figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
    figure.savefig(base_dir + file_name + ".svg", bbox_inches='tight', dpi=400)

def line_plot(df, base_dir, file_name, x_column, y_column, title, hue=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    sns.set(style='whitegrid')
    figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue).set_title(title)
    figure = figure.get_figure()
    figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
    figure.savefig(base_dir + file_name + ".svg", bbox_inches='tight', dpi=400)