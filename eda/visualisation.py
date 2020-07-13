import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

def diagnostic_plot(x, y):
    """
    Derived from diagnostic plot example in Metis Data Science Boot Camp
    """
    plt.figure(figsize=(20,5))
    
    rgr = LinearRegression()
    rgr.fit(x,y)
    pred = rgr.predict(x)

    plt.subplot(1, 3, 1)
    plt.scatter(x,y)
    plt.plot(x, pred, color='blue',linewidth=1)
    plt.title("Regression fit")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.subplot(1, 3, 2)
    res = y - pred
    plt.scatter(pred, res)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")
    
    plt.subplot(1, 3, 3)
    # Generates a probability plot of sample data against the quantiles of a 
    # specified theoretical distribution 
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")

def label_lines_right(lines):
    """
    Take a list containing the lines of a plot (typically the result of 
    calling plt.gca().get_lines()), and add the labels for those lines to the right 
    of the lines.
    [Adapted and simplified from solution to problem: https://stackoverflow.com/q/43573623/4100721]
    """

    # Loop over all lines
    for line in lines:

        # Get basic properties of the current line
        label = line.get_label()
        color = line.get_color()
        x_values = line.get_xdata()
        y_values = line.get_ydata()

        # Short notation for the position of the label
        x = max(x_values)
        y = y_values[x]

        # Actually plot the label onto the line at the calculated position
        plt.text(x, y, label, color=color, horizontalalignment='left', verticalalignment='center',
                 size='large',weight='semibold')

def show_values_on_bars(axs, h_v="v", space=0.4, format='%.1d'):
    """
    Required:
    - axs: the axs to be processed   
    Optional:
    - h_v: 'h' means horizontal bar plot, 'v' means vertical bar plot
    - space: how much space between the bar and the value to be displayed
    - format: how to format the displayed value
    """
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + float(space)
                value = p.get_height()
                ax.text(_x, _y, format % value, ha="center", va="top") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() / 2
                value = p.get_width()
                ax.text(_x, _y, format % value, ha="left", va="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def plot_top_n(source_df, col_to_plot, n, agg_type, category_col, subcategory_col,
               comparison_col=None, category=None, plot_type='total', df_filter=None, show_others=False, 
               **kwargs):
    """
    plots the top n categories in a horizontal bar plot.
    ---
    required args
    - df: the source df
    - col_to_plot: the column to be plotted. must be numeric if agg_type is not count
    - n: number of top categories or subcategories to plot
    - agg_type: mean, sum or count of items. Pass in form {'agg_type': 'label'} to rename the agg_type
    - category_col: the main category column
    - subcategory_col: the subcategory column
    
    optional args
    - comparison_col: segment by this column
    - category: pass a valid category to plot the constituent subcategories
                pass 'All' to plot all subcategories
                pass None to plot main categories
    - plot_type: 'total' shows the raw values; 'percent' shows the percentage of the total
    - df_filter: restrict df
    - show_others: if True, add a bar that aggregartes the values not in the top n
    
    """

    def _df_col_type(df, col):
        return type(df[col].iloc[0])

    # error checking
    _aggtype_error = "invalid agg_type, expected string or dict of form {'agg_type: 'label'}"

    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("invalid source_df, expected a pandas.DataFrame")

    for col_to_check in [col_to_plot, category_col, subcategory_col]:
        if col_to_check not in source_df.columns:
            raise TypeError("expected col_to_plot, category_col and subcategory_col to be column labels of source_df")
    
    if not isinstance(n, int) or n < 1:
        raise TypeError("invalid n, expected integer greater than 0")
    if n > len(source_df):
        raise TypeError("n is larger than length of source_df")

    if isinstance(agg_type, dict) and len(agg_type) == 1:
        col_name = agg_type
        agg_type = next(iter(agg_type)) # returns the key
    elif isinstance(agg_type, str): # default to 'sum', 'mean' or 'count' as the labels
        col_name = {agg_type: agg_type}
    else:
        raise TypeError(_aggtype_error)
    
    if agg_type not in ['sum', 'mean', 'count']:
        raise TypeError(_aggtype_error)

    if not isinstance(_df_col_type(source_df, col_to_plot), (int, float)) and agg_type != 'count':
        raise TypeError("if agg_type is sum or mean, col_to_plot column in source_df must be numeric")

    if isinstance(show_others, bool):
        raise TypeError("invalid others, expected bool")

    if plot_type not in ['total', 'percent']:
        raise TypeError("invalid plot_type, expected 'total' or 'percent'")

    # ordering by mean makes no sense especially if sparse data takes over
    if agg_type == 'mean':
        value_to_order = 'sum'
    else:
        value_to_order = agg_type

    # apply filter to df if needed
    
    if df_filter is not None:
        try:
            source_df = source_df[df_filter]
        except:
            raise TypeError("unable to filter source_df with df_filter")
        
    # set up for overall domain categories or specific category
    if category is not None:
        groupby_col = category_col
        title = f"Domain Categories\n Top {n} by {col_name[agg_type]}"  
    elif category == 'All':
        groupby_col = subcategory_col
        title = f"Domains\n Top {n} by {col_name[agg_type]}"
    elif category in source_df[category_col].values:
        source_df = source_df[source_df[category_col] == category]
        groupby_col = subcategory_col
        title = f"{category}\n Top {n} by {col_name[agg_type]}"
    else:
        raise TypeError("invalid category, expected None or a column label in source_df")
    
    # configure if there is a column to compare with
    if comparison_col is None:
        overall_group = [groupby_col]
    else:
        if comparison_col not in source_df.columns:
            raise TypeError("invalid comparison_col, expected None or a column label in source_df")
        overall_group = [groupby_col, comparison_col]
       
    # configure source df for plotting
    cat_df = \
        source_df \
               .groupby(overall_group)[col_to_plot] \
               .agg(['sum', 'count', 'mean']) \
               .reset_index()
    
    # configure sort order 
    cat_order = \
        source_df \
               .groupby([groupby_col])[col_to_plot] \
               .agg(['sum', 'count', 'mean']) \
               .sort_values(value_to_order, ascending=False) \
               .head(n).index
    cat_order = list(cat_order)
    
    # add others column if needed
    if show_others:
        others_df = source_df \
                .groupby(overall_group)[col_to_plot] \
                .agg(['sum', 'count', 'mean']) \
                .sort_values(value_to_order, ascending=False) \
                .tail(-n)[agg_type]
        
        # need to aggregate the counts by sum, not count
        if agg_type == 'count':
            others_agg = 'sum'
        else:
            others_agg = agg_type
            
        if comparison_col:
            others_df = others_df \
                .groupby(comparison_col) \
                .agg(others_agg)
            
            # create records for 'Others' for each value in comparison col
            for index, value in zip(others_df.index, others_df):
                others_row = {groupby_col: 'Others', 
                              comparison_col: index, 
                              agg_type: value}
                cat_df = cat_df.append(others_row, sort=False, ignore_index=True)
        else:
            # create record for 'Others' straight
            others_row = {groupby_col: 'Others', 
                          agg_type: others_df.agg(others_agg)}
            cat_df = cat_df.append(others_row, sort=False, ignore_index=True)
        
        # add 'Others' to plot order
        cat_order.append('Others')

    # change values to percentage if needed
    if plot_type == 'percent':
        if comparison_col is None:
            cat_df[agg_type] /= cat_df[agg_type].agg(others_agg)
        else:
            new_cat_dflist = []
            for index, value in zip(cat_df[comparison_col].value_counts().index, cat_df[agg_type]):
                comparison_values_df = \
                    (cat_df[cat_df[comparison_col]==index][agg_type] / \
                     cat_df[cat_df[comparison_col]==index][agg_type].agg(others_agg))
                new_cat_dflist.append(comparison_values_df)
            new_cat_df = pd.concat(new_cat_dflist).sort_index()
            cat_df[agg_type] = new_cat_df

    # the actual plot!
    cat_df.rename(columns=col_name, inplace=True)
    ax = sns.barplot(data=cat_df, y=groupby_col, x=col_name[agg_type], 
                     hue=comparison_col, order=cat_order, **kwargs)
    ax.set(title=title, ylabel='')
    
    # configure labels for percentage plot
    if plot_type == 'percent':
        locs, labels = plt.xticks()
        ax.set(title=title + ' percent')
        ax.set_xticklabels([str(int(label * 100)) + '%' for label in locs])
    
    return ax