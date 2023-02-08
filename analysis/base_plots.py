from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.color_palette()

def bar_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, sci=False):
    pass
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    max_value = df[y_column].max()
    fig, ax = plt.subplots()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if sci:
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim([0, 130000])

    figure = sns.barplot(x=x_column, y=y_column, data=df).set_title(title)
    for bars in ax.containers:
        ax.bar_label(bars)
    figure = figure.get_figure()
    figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
    figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def line_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if type is not None:
        palette = sns.color_palette()
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette).set_title(title)
    else:
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax).set_title(title)

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)