from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import numpy as np

# sns.color_palette()

def bar_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, hue_order=None, y_lim=False, y_min=0, y_max=1, log_scale=False, sci=False, x_order=None, ax=None):
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    max_value = df[y_column].max()
    fig, ax = plt.subplots()
    sns.set(style='whitegrid')
    log = ""
    file_name = """{}_barplot""".format(file_name)
    df[y_column] = df[y_column].round(2)
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
    if y_lim:
        # y_max = float(max_value)
        plt.ylim([y_min, y_max])



    figure = sns.barplot(ax=ax, x=x_column, y=y_column, hue=hue, data=df, hue_order=hue_order, errorbar=None, order=x_order).set_title(title)
    for bars in ax.containers:
        ax.bar_label(bars)

    figure = figure.get_figure()
    figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
    figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def box_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, tipo=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None, x_order=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    file_name = """{}_lineplot""".format(file_name)
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # if x_column is not None:
    #     x = df[x_column].tolist()
    #     plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if tipo is not None:
        palette = sns.color_palette()
        figure = sns.boxplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, order=x_order)
    else:
        figure = sns.boxplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, order=x_order)

    figure.set_title(title)

    # if hue is not None:
    #     medians = df.groupby([x_column, hue])[y_column].median()
    # else:
    #     medians = df.groupby([x_column])[y_column].median()
    # print("mediana: ", medians)
    # medians = medians.round(3)
    vertical_offset = df[y_column].median() * 0.05  # offset from median for display

    # for xtick in figure.get_xticks():
    #     figure.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
    #                   horizontalalignment='center', size='x-small', color='w', weight='semibold')

    if tipo == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if tipo == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def violin_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None, x_order=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    file_name = """{}_violinplot""".format(file_name)
    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # if x_column is not None:
    #     x = df[x_column].tolist()
    #     plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if type is not None:
        palette = sns.color_palette()
        figure = sns.violinplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, order=x_order).set_title(title)
    else:
        figure = sns.violinplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, order=x_order).set_title(title)

    if type == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)


def line_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    file_name = """{}_lineplot""".format(file_name)

    if ax is None:
        # fig, ax = plt.subplots()
        figure = plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    x = df[x_column].tolist()
    # plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if type is not None:
        palette = sns.color_palette()
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, style=style).set_title(title)
    else:
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, style=style).set_title(title)
    print("nof")

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if type == 2:
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
        plt.xticks(np.arange(0, max(x)+1, 50))
        plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def stacked_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    file_name = """{}_stacked""".format(file_name)

    if ax is None:
        # fig, ax = plt.subplots()
        figure = plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # x = df[x_column].tolist()
    # # plt.xticks(np.arange(0, max(x) + 1, 2.0))
    #
    labels = df.columns.tolist()
    x = df[x_column].tolist()
    # args =
    # plt.stackplot(x, df[y_column].tolist(), labels=df[hue].tolist())

    (
        so.Plot(df, x_column, y_column, color=hue)
        .add(so.Area(alpha=.1), so.Stack()).save(loc=base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400).save(loc=base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)
    )
    #
    # plt.show()

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if type == 2:
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
        plt.xticks(np.arange(0, max(x)+1, 20))
        plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        # so.Plot().save(loc=base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        # so.Plot().save(loc=base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def hist_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    file_name = """{}_lineplot""".format(file_name)

    if ax is None:
        # fig, ax = plt.subplots()
        figure = plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    x = df[x_column].tolist()
    # plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if type is not None:
        palette = sns.color_palette()
        figure = sns.histplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, multiple="stack").set_title(title)
    else:
        figure = sns.histplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, multiple="stack").set_title(title)
    print("nof")

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if type == 2:
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
        plt.xticks(np.arange(0, max(x)+1, 20))
        plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def heatmap_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None):
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
        figure = sns.heatmap(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, style=hue).set_title(title)
    else:
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, style=hue).set_title(title)

    if type == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def ecdf_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None):
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
        figure = sns.ecdfplot(x=x_column, data=df, hue=hue, ax=ax, palette=palette).set_title(title)
    else:
        figure = sns.ecdfplot(x=x_column, data=df, hue=hue, hue_order=hue_order, ax=ax).set_title(title)

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)