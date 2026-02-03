import duckdb, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# RYB palette, dark to light
reds = ['#912322', '#c23637', '#d95d5b', '#ea9a9d', '#f9c9c7']
blues = ['#004586', '#0068a9', '#4d8fcb', '#92c4e9', '#c1e4f4']
yellows = ['#926c17', '#c59527', '#e8c047', '#f5d77f', '#ffefbb']
purples = ['#6e2769', '#b271ab', '#cda0cb', '#e8d0e6']
oranges = ['#ae460b', '#e96302', '#f48f3d', '#fbb874', '#fddbb5']
overleaf_background = '#FFFFFF' # Pure white
font_colour = 'black'
green_hex = '#80C080'
grey_hex = '#C0C0C0'

def style_nature(column='single'):
    
    # Set figure width. 90mm for single column. 180 for double
    # matplotlib sets figure size in inches

    if column == 'double':
        fig_width_mm = 180  
    else:
        fig_width_mm = 90   
        
    mm_to_inches = 0.0393701
    fig_width = fig_width_mm * mm_to_inches
    
    # Set dpi to minimum requirement of 300 dpi. plt defaults to 100.
    dpi = 300.0
    
    # Set linewidths
    linewidth = 0.5
    
    # Set fonts. Nature requires Arial or Helvetica
    # font = '' # Arial or Helvetica
    fontsize_small = 5 #pt
    fontsize_large = 7 #pt
      
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.family'] = 'sans-serif'

    mpl.rcParams['font.size'] = fontsize_large
    mpl.rcParams['axes.labelsize'] = fontsize_large
    mpl.rcParams['axes.titlesize'] = fontsize_large
    mpl.rcParams['xtick.labelsize'] = fontsize_large
    mpl.rcParams['ytick.labelsize'] = fontsize_large

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
      
    return fig_width, dpi, linewidth

def set_colour_style(font, background):
    
    # Set background colour
    plt.rcParams['axes.facecolor'] = background
    plt.rcParams['figure.facecolor'] = background
    # Set font colour
    plt.rcParams['text.color'] = font
    plt.rcParams['axes.labelcolor'] = font
    plt.rcParams['xtick.color'] = font
    plt.rcParams['ytick.color'] = font
    
    return

def get_table_1(db_pth, pdf_in, adf, is_all_data=True, is_asset_data=True):
    from dateutil.relativedelta import relativedelta
    
    def format_date(s):
        for c in ['mean','min','25%','50%','75%','90%','99%','max']:
            s.loc[c] = pd.to_datetime(s.loc[c]).strftime('%m/%y')
        return s
    
    def merge_in_users_data_and_get_month_stats(df_in):       
        with duckdb.connect(db_pth) as con:
            users = con.execute("""SELECT CAST(id AS INTEGER) AS user_id, 
                                LAST(age) AS age, 
                                LAST(gender) AS gender FROM users 
                                GROUP BY user_id""").df()
            users['user_id'] = users['user_id'].astype(int)

        users['is_female'] = (users['gender']=='Female')*1
        df = pd.merge(df_in, users, on='user_id', how='left')
    
        # Timing stats
        df['month'] = pd.to_datetime(df['month'])
        userlevel = df.groupby('user_id').agg({'month':['min','max'], 'age':['first'], 'is_female':['first']})
        userlevel.columns = ['min_month', 'max_month', 'age', 'is_female']
        userlevel['months_in_data'] = userlevel.apply(
            lambda row: relativedelta(row['max_month'], row['min_month']).years * 12 +
                        relativedelta(row['max_month'], row['min_month']).months,
            axis=1
        )
        return pd.merge(df, userlevel, left_on='user_id', right_index=True), userlevel
    
    # Get count and sum of gambling transactions and overall count and sum into pdf
    trans_count = adf.groupby(['user_id', 'month'])[['sum_of_spends','count_of_spends']].sum().reset_index()
    trans_count.columns = ['user_id', 'month', 'sum_of_spends', 'count_of_spends']
    trans_count['sum_of_spends'] = -trans_count['sum_of_spends']
    pdf = pd.merge(pdf_in, trans_count[['user_id', 'month', 'count_of_spends']], how='left', on=['user_id', 'month'])
    
    gamb_trans_count = adf[adf.COICOP==14.0].groupby(['user_id', 'month'])[['sum_of_spends','count_of_spends']].sum().reset_index()
    gamb_trans_count = pd.merge(adf[['user_id', 'month']].drop_duplicates(), gamb_trans_count, how='left', on=['user_id', 'month']).fillna(0) # Ensure we have months with no gambling
    gamb_trans_count.columns = ['user_id', 'month', 'sum_of_gambling_spends', 'gambling_transactions']
    gamb_trans_count['sum_of_gambling_spends'] = -gamb_trans_count['sum_of_gambling_spends']
    pdf = pd.merge(pdf, gamb_trans_count[['user_id', 'month',  'gambling_transactions']], how='left', on=['user_id', 'month']).fillna(0)
    
    # Initialise output
    out_df = pd.DataFrame()

    if is_all_data:       
        ## First do overall stats (before creation of pdf asset)
        df, userlevel = merge_in_users_data_and_get_month_stats(adf)
            
        # User level
        s = userlevel.age.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_age':s})], axis=1) 
        
        s = userlevel.is_female.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_is_female':s})], axis=1) 
        
        s = userlevel.min_month.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_min_month':format_date(s)})], axis=1)
        
        s = userlevel.max_month.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_max_month':format_date(s)})], axis=1)
        
        s = userlevel.months_in_data.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_months_in_data':s})], axis=1) 
    
        # Transaction count stats
        s = trans_count.groupby('user_id').count_of_spends.sum().describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_user_total_transaction_count':s})], axis=1)
        
        s = gamb_trans_count.groupby('user_id').gambling_transactions.sum().describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_user_gambling_transaction_count':s})], axis=1)    

        # Spending stats
        s = trans_count.groupby('user_id').sum_of_spends.sum().describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_user_total_spend':s})], axis=1) 
        
        s = gamb_trans_count.groupby('user_id').sum_of_gambling_spends.sum().describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'overall_user_gambling_spend':s})], axis=1) 


    if is_asset_data:
        ## Now stats about the data asset
        df, userlevel = merge_in_users_data_and_get_month_stats(pdf)    
        
        s = userlevel.age.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_age':s})], axis=1) 
        
        s = userlevel.is_female.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_is_female':s})], axis=1) 
        
        s = userlevel.min_month.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_min_month':format_date(s)})], axis=1)
        
        s = userlevel.max_month.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_max_month':format_date(s)})], axis=1)
        
        s = userlevel.months_in_data.describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_months_in_data':s})], axis=1) 
        
        s = df['gambling_transactions'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_month_gambling_transaction_count':s})], axis=1)
    
        s = df['count_of_spends'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_month_total_transaction_count':s})], axis=1)    
            
        s = df['COICOP_14.0'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_month_gamb_spend':s})], axis=1) 
        
        s = df['total_spend'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_user_month_total_spend':s})], axis=1) 
            
        # # gambling user_month level
        df, userlevel = merge_in_users_data_and_get_month_stats(pdf[pdf['decile_abs']!=0.0]) 
        
        s = df['gambling_transactions'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_gambling_user_month_gambling_transaction_count':s})], axis=1)
    
        s = df['count_of_spends'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_gambling_user_month_total_transaction_count':s})], axis=1)    
            
        s = df['COICOP_14.0'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_gambling_user_month_gamb_spend':s})], axis=1) 
        
        s = df['total_spend'].describe(percentiles=[.25,.50,.75,.90,.99])
        out_df = pd.concat([out_df, pd.DataFrame({'asset_gambling_user_month_total_spend':s})], axis=1) 
    
    out_df = out_df.T
    
    # Ensure integer values where appropriate
    rows = ['overall_age',
        'overall_is_female',
        'overall_months_in_data',
        'overall_user_total_transaction_count',
        'overall_user_gambling_transaction_count',
        'asset_user_age',
        'asset_user_is_female',
        'asset_user_months_in_data',
        'asset_user_month_gambling_transaction_count',
        'asset_user_month_total_transaction_count',
        'asset_gambling_user_month_gambling_transaction_count',
        'asset_gambling_user_month_total_transaction_count']
    cols = ['min', '25%', '50%', '75%', '90%', '99%', 'max']
    rows = [r for r in rows if r in out_df.index ]
    out_df.loc[rows, cols] = out_df.loc[rows, cols].astype(int)
    out_df['count'] = out_df['count'].astype(int)
    
    # Reduce the output width
    out_df = out_df[['count', 'mean', 'std', '25%', '50%', '75%', '90%', '99%']]
    out_df.columns = ['N', 'Mean', 's.d.', '25th', '50th', '75th', '90th', '99th']
        
    return out_df


def get_overall_demographics_to_latex():
    
    # US to start with
    from locations import db_pth_us, pop_pdf_pth_us, pop_asset_pq_us, asset_pq_us, pdf_pth_us, output_folder
    pdf_pop_us = pd.read_parquet(pop_pdf_pth_us)
    adf_pop_us = pd.read_parquet(pop_asset_pq_us)
    us_population_data_descriptives = get_table_1(db_pth_us, pdf_pop_us, adf_pop_us, is_all_data=True, is_asset_data=False)

    pdf_us = pd.read_parquet(pdf_pth_us)
    adf_us = pd.read_parquet(asset_pq_us)
    us_asset_data_descriptives = get_table_1(db_pth_us, pdf_us, adf_us, is_all_data=False, is_asset_data=True)
    us_table_1 = pd.concat([us_population_data_descriptives,us_asset_data_descriptives],axis=0)

    ltx_str = us_table_1.to_latex(index=True,
                        column_format="lcccccccccc",
                        multicolumn=True, multicolumn_format='c',         
                        float_format="{:.2f}".format,  
                        na_rep="--",                       
                        escape=False,                      
                        bold_rows=False,                   
                        header=True).replace('%', '\\%').replace('_', '\\_')

    ltx_out = f'{output_folder}/us_overall_demographics.tex'
    with open(ltx_out, 'w') as f:
            f.write(ltx_str)
            print('Output to : ', ltx_out)
                
    # Now GB
    from locations import db_pth_uk, pop_pdf_pth_uk, pop_asset_pq_uk, asset_pq_uk, pdf_pth_uk
    pdf_pop_gb = pd.read_parquet(pop_pdf_pth_uk)
    adf_pop_gb = pd.read_parquet(pop_asset_pq_uk)
    gb_population_data_descriptives = get_table_1(db_pth_uk, pdf_pop_gb, adf_pop_gb, is_all_data=True, is_asset_data=False)

    pdf_gb = pd.read_parquet(pdf_pth_uk)
    adf_gb = pd.read_parquet(asset_pq_uk)
    gb_asset_data_descriptives = get_table_1(db_pth_uk, pdf_gb, adf_gb, is_all_data=False, is_asset_data=True)

    uk_table_1 = pd.concat([gb_population_data_descriptives,gb_asset_data_descriptives],axis=0)
    ltx_str = uk_table_1.to_latex(index=True,
                        column_format="lcccccccccc",
                        multicolumn=True, multicolumn_format='c',         
                        float_format="{:.2f}".format,  
                        na_rep="--",                       
                        escape=False,                      
                        bold_rows=False,                   
                        header=True).replace('%', '\\%').replace('_', '\\_')

    ltx_out = f'{output_folder}/uk_overall_demographics.tex'
    with open(ltx_out, 'w') as f:
            f.write(ltx_str)
            print('Output to : ', ltx_out) 

    return us_table_1, uk_table_1

def get_decile_width_stats(df_in, max_width):
    '''
    For the stacked bar plots

    This enables is to ensure that the area each block in the stacked bar is proportional 
    to the spend. We do this by calculating a column width proportional to
    the spend per person in that block
    '''
    df = pd.DataFrame()
    df['spend'] = df_in.groupby('decile')['Total spend'].sum()
    
    udf = df_in[['decile', 'user_id', 'month']].drop_duplicates()
    df['count'] = udf.groupby('decile').decile.count()
    
    df['av_spend'] = df['spend'] / df['count']  
    df['width'] = (df['av_spend'] / df['av_spend'].max()) * max_width
    return df


def order_reduce_data(absmdf, vert_cols,
                      remove_col, 
                      dic_COICOP_short, 
                      combine_tiny, combine_tiny_size, 
                      ordering
                      ):

    # Get pivoted dataframe
    pivot_df = absmdf.pivot_table(index='decile', columns='COICOP', values='Total spend', aggfunc='sum', fill_value=0)
    pivot_df = pivot_df[vert_cols]
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0).sort_index(ascending=True) # Divide through so it adds up to 1
    
    # Process categories to remove small proportions and order categories
    decile_zero = pivot_df.loc[0]

    # Is remove_col included in passed dataframe?
    if (remove_col in decile_zero.index.tolist()):
        inc_remove_col = True
    else:
        inc_remove_col = False

    if (combine_tiny):
        # Identify categories in 0th decile smaller than a specified size
        small_categories = decile_zero[decile_zero<=combine_tiny_size]
        small_categories_labels = small_categories.index.tolist()
        # Remove 'remove_col' from this list (0% by definition)
        small_categories_labels = [category for category in small_categories_labels if (remove_col not in category)]
        small_categories = small_categories[small_categories_labels]
        
        # Check categories remain small across deciles
        small_always = [np.all(pivot_df[category]<combine_tiny_size) for category in small_categories_labels]
        small_categories = small_categories[small_always]
        small_categories_labels = small_categories.index.tolist()
       
        # Order small categories by size, small-large vertically in legend as in main legend
        small_ordered_idx = np.array(np.argsort(small_categories))
        small_categories = [small_categories_labels[i] for i in small_ordered_idx]
        
        # Combine tiny categories and add to dataframe
        small_df = pivot_df[small_categories]
        combined_col = small_df.sum(axis='columns')
        combined_name = 'Combined_COICOP'
        
        pivot_df[combined_name] = combined_col
        dic_COICOP_short[combined_name] = f'Categories with\n< {combine_tiny_size*100:.1f}% spend'
        
        # Redefine decile_zero to only include categories greather than specified size (not remove_col)
        categories_exclude = small_categories + [remove_col]
        categories_include = [category for category in decile_zero.index.tolist() if category not in categories_exclude]
        decile_zero = decile_zero[categories_include]
        

    # Get column names after small proportions removed
    cols = decile_zero.index.tolist()
    
    # Order categories
    if (ordering!='none'):
        
        no_remove_col = decile_zero[decile_zero.index!=remove_col] # 0th decile
        
        if (ordering=='ascending'):
            cols_ordered_idx = np.array(np.argsort(no_remove_col))
            
        elif (ordering=='descending'):
            cols_ordered_idx = np.array(np.argsort(-no_remove_col))
            
        cols_ordered = [cols[i] for i in cols_ordered_idx]
        
    else:
        cols_ordered = cols
    
    # Add combined column
    if (combine_tiny):
        cols_ordered = np.append(cols_ordered, combined_name)
    # Add 'remove_col' to end if required
    if (inc_remove_col):
        cols_ordered = np.append(cols_ordered, remove_col)
        
    # Reorder dataframe and remove combined columns
    pivot_df = pivot_df[cols_ordered]
    
    return pivot_df, dic_COICOP_short, small_categories

def plot_area_bar_plot(absmdf, 
                       COICOP_labels_df, 
                       vert_cols,
                       remove_col,
                       xlabel,
                       ylabel,
                       dic_colours,
                       dic_COICOP_short, 
                       combine_tiny_size=0.01,
                       ax=None,
                       reorder_col=None):
    """
    Draws the area/bar stack on the provided Matplotlib axis.
    Returns (dic_colours, pivot_df, width_stats, dic_COICOP_short, small_categories).
    """

    import numpy as np
    from matplotlib.ticker import PercentFormatter

    assert ax is not None, "Please pass an axis (ax=...) to plot_area_bar_plot"

    # --- inputs / local opts ---
    bar_width_labels = False
    bar_width_labels_angle = 90
    ordering = 'descending'  # 'ascending', 'descending', 'none'

    # --- compute widths, order, reduction ---
    max_width = 1.0
    width_stats = get_decile_width_stats(absmdf[absmdf.COICOP.isin(vert_cols)], max_width)

    pivot_df, dic_COICOP_short, small_categories = order_reduce_data(
        absmdf, vert_cols, remove_col, dic_COICOP_short,
        True, combine_tiny_size, ordering
    )

    # positions / widths
    widths = np.array(width_stats['width'])
    left_edges = np.zeros_like(widths)
    left_edges[1:] = np.cumsum(widths)[:-1]
    bottom = np.zeros(len(pivot_df))
    xtick_positions = left_edges + widths * 0.5

    # outlines
    _, _, linewidth_nature = style_nature()
    bar_outline_weight = linewidth_nature
    bar_outline_colour = overleaf_background

    # init palette if not provided
    if not dic_colours:
        colour_palette = reds + blues + yellows + purples
        dic_colours = {col: colour_palette[i] for i, col in enumerate(pivot_df.columns)}

    # Reorder columns if a new order is passed
    if reorder_col:
        col_order = [c for c in reorder_col if c in pivot_df.columns]
        col_order.reverse()
        pivot_df = pivot_df[col_order]        

    # draw stacks
    for column in pivot_df.columns:
        ax.bar(
            left_edges, pivot_df[column],
            align='edge', width=widths, bottom=bottom,
            color=dic_colours[column], label=dic_COICOP_short[column],
            linewidth=bar_outline_weight, edgecolor=bar_outline_colour
        )
        bottom += pivot_df[column].values

    # x axis
    ax.set_xlim(0.0, float(np.sum(widths)))
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(pivot_df.index)
    if xlabel:
        ax.set_xlabel(xlabel)
    if not bar_width_labels:
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    # y axis (percentage 0–100%)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if ylabel:
        ax.set_ylabel('Proportion of monthly spend')

    # tidy spines
    for s in ax.spines.values():
        s.set_visible(False)

    return dic_colours, pivot_df, width_stats, dic_COICOP_short, small_categories

def get_figure_2(pdf, pdf_us, pdf_coffee):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    
    from locations import output_folder
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import get_labels_and_colour_dics
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dic_COICOP_short, dic_COICOP_colours = get_labels_and_colour_dics(COICOP_labels_df)
    decile_col = 'decile_abs' 

    # --- overall figure sizing & style ---
    fig_width, dpi_nature, linewidth_nature = style_nature(column='double')
    fig_height = (fig_width/2.2) # Ratio of 2.2
    
    print(fig_width, fig_height)
    set_colour_style(font_colour, overleaf_background)
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_nature, constrained_layout=True)
    axes.axis('off')
    
    # Now split so that we have plots at the top and legend at the bottom
    gs = gridspec.GridSpec(
        nrows=2, ncols=1, figure=fig,
        wspace=0.05,
        height_ratios=[(fig_width/3), ((fig_width/2.2)-(fig_width/3))]
    )
    
    # Split top into three
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0], hspace=0.05, width_ratios=[1, 1, 1]
    )
        
    ax_uk = fig.add_subplot(gs_top[0]) 
    ax_us = fig.add_subplot(gs_top[1]) 
    ax_coffee = fig.add_subplot(gs_top[2]) 
    ax_legend = fig.add_subplot(gs[1])     
    ax_legend.axis('off')

    combined_columns = [
    'COICOP_14.0','COICOP_16.0','Combined_COICOP','COICOP_15.0', 'COICOP_10.0','COICOP_6.0','COICOP_3.0',
    'COICOP_12.1','COICOP_5.0','COICOP_9.0','COICOP_13.2','COICOP_12.25',
    'COICOP_8.0','COICOP_12.23','COICOP_11.0','COICOP_4.0','COICOP_7.0',
    'COICOP_13.1','COICOP_12.22','COICOP_1.0'
    ]

    # --- Top left: (a) Gambling ---
    remove_col = 'COICOP_14.0'
    COICOP_cols = [c for c in pdf.columns if (('COICOP' in c) and (c != remove_col))]
    cols = ['user_id', 'month'] + COICOP_cols + [remove_col, decile_col]
    gambling_long = pdf[cols].melt(
        id_vars=[decile_col, 'user_id', 'month'],
        var_name='COICOP',
        value_name='Total spend'
    ).rename(columns={decile_col: 'decile'})

    xlabel = 'Gambling spend decile'
    (dic_colours, pivot_df_gambling, width_stats_g, dic_COICOP_short, small_categories_g) = plot_area_bar_plot(
        gambling_long, COICOP_labels_df,
        COICOP_cols + [remove_col],
        remove_col,
        xlabel=xlabel,
        ylabel=True,     # common y on the left panel
        dic_colours=dic_COICOP_colours,
        dic_COICOP_short=dic_COICOP_short,
        ax=ax_uk, reorder_col=combined_columns
    )


    # --- Top right: (b) US data ---
    remove_col = 'COICOP_14.0'
    COICOP_cols = [c for c in pdf_us.columns if (('COICOP' in c) and (c != remove_col))]
    cols = ['user_id', 'month'] + COICOP_cols + [remove_col, decile_col]
    gambling_long_us = pdf_us[cols].melt(
        id_vars=[decile_col, 'user_id', 'month'],
        var_name='COICOP',
        value_name='Total spend'
    ).rename(columns={decile_col: 'decile'})

    (dic_colours, us_pivot_df_gambling, us_width_stats_g, dic_COICOP_short, us_small_categories_g) = plot_area_bar_plot(
        gambling_long_us, COICOP_labels_df,
        COICOP_cols + [remove_col],
        remove_col,
        xlabel=xlabel,
        ylabel=False,     # common y on the left panel
        dic_colours=dic_COICOP_colours,
        dic_COICOP_short=dic_COICOP_short,
        ax=ax_us, reorder_col=combined_columns
    )
    ax_us.set_yticklabels([])

    # --- Bottom left: (c) Coffee deciles ---
    remove_col = 'COICOP_16.0'
    xlabel = 'Coffee shop spend decile'
    COICOP_cols_c = [c for c in pdf_coffee.columns if ('COICOP' in c)]
    cols_c = ['user_id', 'month'] + COICOP_cols_c + [decile_col]
    coffee_long = pdf_coffee[cols_c].melt(
        id_vars=[decile_col, 'user_id', 'month'],
        var_name='COICOP',
        value_name='Total spend'
    ).rename(columns={decile_col: 'decile'})


    # Put coffee on top
    combined_columns_coffee = [
    'COICOP_16.0','COICOP_14.0','Combined_COICOP','COICOP_15.0', 'COICOP_10.0','COICOP_6.0','COICOP_3.0',
    'COICOP_12.1','COICOP_5.0','COICOP_9.0','COICOP_13.2','COICOP_12.25',
    'COICOP_8.0','COICOP_12.23','COICOP_11.0','COICOP_4.0','COICOP_7.0',
    'COICOP_13.1','COICOP_12.22','COICOP_1.0'
    ]

    (dic_colours, pivot_df_coffee, width_stats_c, dic_COICOP_short, small_categories_c) = plot_area_bar_plot(
        coffee_long, COICOP_labels_df,
        COICOP_cols_c,
        remove_col,
        xlabel=xlabel,
        ylabel=False,
        dic_colours=dic_COICOP_colours,
        dic_COICOP_short=dic_COICOP_short,
        ax=ax_coffee, reorder_col=combined_columns_coffee
    )
    ax_coffee.set_yticklabels([])
    ax_coffee.set_xlabel('Coffee spend decile')
    
    # Confirm y-limits
    for a in (ax_uk, ax_us, ax_coffee):
        a.set_ylim(0, 1.0)

    # Now do panel labels.
    ax_uk.text(-0.24, 1.0, 'a',
                    transform=ax_uk.transAxes,
                    ha='left', va='top', fontweight='bold')

    ax_us.text(-0.1, 1.0, 'b',
                    transform=ax_us.transAxes,
                    ha='left', va='top', fontweight='bold')

    ax_coffee.text(-0.1, 1.0, 'c',
                    transform=ax_coffee.transAxes,
                    ha='left', va='top', fontweight='bold')

    # # --- Bottom section: Common legend in specified order ---

    legend_items = []
    dic['Combined_COICOP'] = f'Categories with\n<1% spend'
    for code in combined_columns:
        # skip if colour not present (e.g., category absent in some panels)
        if code in dic_colours and code in dic:
            c = dic_colours[code]
            label = dic_COICOP_short[code]
            legend_items.append(mpatches.Patch(facecolor=c, edgecolor='none', label=label))

    leg = ax_legend.legend(
        handles=legend_items,
        loc='center',#loc='upper left',
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        ncol=7
    )

    # --- save to pdf ---
    saveto = f'{output_folder}/figure_2.pdf'
    fig.savefig(saveto, format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.00)
    print("Saved:", saveto)    
    
    
    
def double_barplot_of_regression_coefficients(
    gb_ldf, us_ldf, 
    model, 
    effect_variable,
    dic_colours, 
    dic_COICOP_short,
    save_path,
    ylim):

    # --- Figure style and sizing ---
    fig_width, dpi_nature, linewidth_nature = style_nature(column='double')
    fig_width = fig_width*0.975
    
    # Keep panel height proportional to a *single* column width
    wspace=0.2 # Space between panels
    single_col_width = (fig_width-wspace) / 2.0
    fig_height = single_col_width * 0.7

    # Colours, background, etc.
    set_colour_style(font_colour, overleaf_background)

    # --- Filter data for each market ---
    def prep_df(ldf):
        df = ldf[(ldf['Variable'] == effect_variable) &
                 (ldf['Model'] == model)][
                 ['Variable', 'Coef.', 'CI Lower' , 'CI Upper', 'Std.Err.', 'Dep. variable code', 'N']
               ].copy()
                 
        # Sort by coefficient ascending
        df = df.sort_values(by='Coef.', ascending=True)

        # Grab N (assumes unique)
        n_vals = df['N'].dropna().unique()
        N_val = n_vals[0] if len(n_vals) else None
        return df, N_val

    gb_df, N_gb = prep_df(gb_ldf)
    us_df, N_us = prep_df(us_ldf)

    # Ensure a consistent category order across both panels (use GB order)
    order = gb_df['Dep. variable code'].tolist()
    us_df = (us_df
             .set_index('Dep. variable code')
             .reindex(order)  
             .reset_index())

    # Compute global y-limits if not provided
    y_min, y_max = ylim
    
    # --- Build figure with two axes ---
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), dpi=dpi_nature, sharey=False
    )
    fig.patch.set_alpha(0.0)  # transparent figure bg

    # Common error-bar style
    errorbars_format = {
        'ecolor': font_colour,
        'elinewidth': linewidth_nature,
        'capsize': 0,
        'capthick': linewidth_nature
    }

    def panel(ax, df, market, panel_letter, N_val):
        # Colors by category
        df = df.sort_values('Coef.')
        ccodes = [dic_colours[c] for c in df['Dep. variable code']]
        yerr = np.vstack([
            df['Coef.'] - df['CI Lower'],  
            df['CI Upper'] - df['Coef.']   
        ])
        bars = ax.bar(
            df['Dep. variable code'],
            df['Coef.'],
            color=ccodes,
            yerr=yerr,
            error_kw=errorbars_format
        )
        

        # # Zero line per-axes
        # ax.axhline(0.0, color='k', linestyle='--', linewidth=linewidth_nature)

        # Margins
        ax.set_xmargin(0.025)
        ax.set_ymargin(0.025)
        ax.set_ylim(y_min, y_max)

        # X tick labels (top)
        xtick_labels = [dic_COICOP_short[k] for k in df['Dep. variable code'].tolist()]
        for i, c in enumerate(xtick_labels):
            if 'cohol' in c:
                xtick_labels[i] = 'Alcohol, etc'

        # Keep existing tick positions; move labels to the top
        ax.set_xticks(ax.get_xticks())  # ensure we can set labels explicitly
        ax.set_xticklabels(
            xtick_labels,
            rotation=90,
            rotation_mode='anchor',
            ha='left',   # works better at the top; adjust if you prefer
            va='center'
        )
        ax.tick_params(axis='x', bottom=False, labelbottom=False,
                       top=True, labeltop=True, length=0)

        # Remove all spines (no box)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('none')

        # Panel tag and N (8 pt bold, upright)
        country = 'UK' if market == 'GB' else 'US'
        tag_text = f'({panel_letter}) {country}. N = {N_val}' if N_val is not None else f'({panel_letter}) {country}'
                        
        tick_values = [-0.50, -0.40, -0.30, -0.20, -0.10, 0.00]
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_locator(mticker.FixedLocator(tick_values))
        
        if market == 'US': 
            ax.set_yticklabels(['-$0.50','-$0.40', '-$0.30', '-$0.20', '-$0.10', '$0.00'])
            panel_letter='b'
        elif market=='GB':
            ax.set_yticklabels(['-£0.50', '-£0.40', '-£0.30', '-£0.20', '-£0.10', '£0.00'])
            panel_letter='a'
            
        
        ax.text(
            -0.1, 1.4, panel_letter,  # position in axes coords
            transform=ax.transAxes,
            fontsize=7,
            fontweight='bold',
            va='top', ha='left'
        )

    # Left = GB (panel a); Right = US (panel b)
    panel(axes[0], gb_df, 'GB', 'a', N_gb)
    panel(axes[1], us_df, 'US', 'b', N_us)
    axes[0].tick_params(axis='y', labelleft=True, labelright=False, length=0)
    axes[1].tick_params(axis='y', labelleft=True, labelright=False, length=0)
    # axes[1].tick_params(axis='y', labelleft=False, labelright=True, length=0)

    # # Layout:
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.05, top=0.8, wspace=wspace)    
    fig.set_size_inches(fig_width, fig_height, forward=True)
    
    print(f'Saving figure to {save_path}')
        
    tmp =plt.rcParams['figure.autolayout'] 
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.autolayout'] = tmp
    fig.savefig(save_path, format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)


def get_figure_3():
    from locations import output_folder
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import combinations, get_labels_and_colour_dics
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dicShort, colour_dic = get_labels_and_colour_dics(COICOP_labels_df)
    
    lbl, control_variables, net_or_gross = combinations[0]

    gb_pth = f'{output_folder}/UK_regression_results_{net_or_gross}-control_by_{lbl}.xlsx'
    gb_ldf = pd.read_excel(gb_pth, index_col='Unnamed: 0')
    print("Reading : ", gb_pth)

    us_pth = f'{output_folder}/US_regression_results_{net_or_gross}-control_by_{lbl}.xlsx'
    us_ldf = pd.read_excel(us_pth, index_col='Unnamed: 0')
    print("Reading : ", us_pth)

    ylim=[-0.47, 0.0]
    effect_variable = 'COICOP_14.0'
    model = 'Fixed effects'
    save_path =  f'{output_folder}/figure_3.pdf'

    double_barplot_of_regression_coefficients(
        gb_ldf, us_ldf, 
        model, 
        effect_variable,
        colour_dic, 
        dicShort,
        save_path,
        ylim=ylim)

def plot_classification_proportion(db_pth, out_pth, class_prop=20000):
    '''Plot the proportion of total debits that are classified by ChatGPT
    class_prop is the top # of brands that were classified, ordered by
    total debit spend
    '''    

    with duckdb.connect(db_pth) as con:   
        brand_debit_counts = con.execute("""SELECT brand, COUNT(*) AS cnt, -SUM(amount) AS tot FROM transactions WHERE amount<0.0 AND brand<>'' GROUP BY brand ORDER BY tot DESC""").df()
    brand_debit_counts['openai_classified'] = (brand_debit_counts.index<class_prop)
    brand_debit_counts['rank'] = brand_debit_counts.index
    brand_debit_counts['cum_sum_debits'] = brand_debit_counts['tot'].cumsum()
    stats = brand_debit_counts.groupby('openai_classified').agg({'cnt':'sum', 'tot':'sum', 'brand':'count'}).reset_index()

    for c in ['cnt', 'tot', 'brand']:
        sm = stats[c].sum()
        stats[f'prop_{c}'] = stats[c].apply(lambda x: f'{(100.0*x/sm):.1f}%') #stats['tot']/stats['tot'].sum()
    stats = stats.set_index('openai_classified', drop=True)
    stats.columns = ['transaction_count','sum_of_debits','count_of_brands', 'percentage_of_transactions', 'percentage_of_debits', 'percentage_of_brands']
    
    # Plot cumulative spend line
    fig_width, dpi_nature, linewidth_nature = style_nature()

    fig, ax = plt.subplots(figsize=(fig_width, fig_width), dpi=dpi_nature)
    plt_df = brand_debit_counts[:(2*class_prop)]
    ax.plot(plt_df['rank'], plt_df['cum_sum_debits'], color='white', linewidth=linewidth_nature)
    total_spend = brand_debit_counts['tot'].sum()

    # Unclassified
    grey_fill = ax.fill_between(
        plt_df['rank'],
        plt_df['cum_sum_debits'],
        where=plt_df['rank'] >= class_prop,
        color=grey_hex,
        alpha=1.0
    )
    ax.text(
        x=24000,  
        y=total_spend * 0.91, 
        s='Unclassified or classified\nby supplier algorithm',
        color=grey_hex,
        fontsize=7
    )

    # Classified
    green_fill = ax.fill_between(
        plt_df['rank'],
        plt_df['cum_sum_debits'],
        where=plt_df['rank'] < class_prop,
        color=green_hex,
        alpha=1.0
    )    
    y_max = plt_df.loc[plt_df['rank'] == 20000, 'cum_sum_debits'].values[0]
    ax.fill_between( # and the part to the right
        x=[20000, 40000],
        y1=0,
        y2=y_max,
        color=green_hex,
        alpha=1.0
    )
    ax.text(
        x=12500,  
        y=total_spend * 0.4,  
        s='Classified by o4-mini',
        color='white',
        fontsize=7
    )

    # Add horizontal dashed red line at total spend
    ax.axhline(y=total_spend, color='black', linestyle='--', label='Total Spend')
    ax.text(
        x=-1000, 
        y=total_spend * 0.96,  
        s='Total debits',
        color='black',
        fontsize=7
    )

    ax.set_xlabel('Brand rank by total debit spend')
    ax.set_ylabel('Cumulative spend (billions of £)')
    
    
    tick_values = [v*10**9 for v in range(6)]
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_values))    
    ax.set_yticklabels(['0','1','2','3','4','5' ])

    plt.tight_layout()
    # plt.show()
    ax.get_figure().savefig(out_pth, format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    print(stats.transpose())
    

def get_decile_demographic_data(pdf_in, db_pth, dicShort, latex_out, is_coffee, deb_cred_df=pd.DataFrame()):
        
    with duckdb.connect(db_pth) as con:
        users = con.execute("""SELECT CAST(id AS INTEGER) AS user_id, 
                            LAST(age) AS age, 
                            LAST(gender) AS gender FROM users 
                            GROUP BY user_id""").df()
        users['user_id'] = users['user_id'].astype(int)

    users['is_female'] = (users['gender']=='Female')*1
    df = pd.merge(pdf_in, users, on='user_id', how='left')
    
    if not is_coffee:
        gamb_debits = deb_cred_df[deb_cred_df.is_debit & (deb_cred_df.COICOP==14.0)]
        df = pd.merge(df, gamb_debits[['user_id', 'month', 'sum']].rename(columns={'sum':'gamb_debits'}), on=['user_id', 'month'], how='left')  
        df['gamb_debits'] = df['gamb_debits'].fillna(0)
    
    out_df = pd.DataFrame()
    
    ## Run stats       
    s = df.groupby('decile_abs')['age'].mean()
    out_df = pd.concat([out_df, pd.DataFrame({'Age':s})], axis=1)

    s = df.groupby('decile_abs')['is_female'].mean()
    out_df = pd.concat([out_df, pd.DataFrame({'Prop. female':s})], axis=1)    
    
    if is_coffee:
        spend_col = 'COICOP_16.0'
        lbl = 'Coffee'
        s = df.groupby('decile_abs')[spend_col].mean()
        out_df = pd.concat([out_df, pd.DataFrame({f'Gross {lbl} Spend':s})], axis=1) 
    
    else:
        spend_col = 'COICOP_14.0'
        lbl = 'Gambling'
        s = df.groupby('decile_abs')['gamb_debits'].mean()
        out_df = pd.concat([out_df, pd.DataFrame({f'Gross {lbl} Spend':s})], axis=1) 
        
        s = df.groupby('decile_abs')[spend_col].mean()
        out_df = pd.concat([out_df, pd.DataFrame({f'Net {lbl} Spend':s})], axis=1)         
   
    
    s = df.groupby('decile_abs')['total_spend'].mean()/1000
    out_df = pd.concat([out_df, pd.DataFrame({f'Total Monthly Spend (thousands)':s})], axis=1)
    
    df['sp_ex_cat'] = df['total_spend'] - df[spend_col]
    s = df.groupby('decile_abs')['sp_ex_cat'].mean()/1000
    out_df = pd.concat([out_df, pd.DataFrame({f'Total Monthly Spend excl {lbl} (thousands)':s})], axis=1)
    
    df['prop_spend_cat'] = df[spend_col]/df['total_spend'] 
    s = df.groupby('decile_abs')['prop_spend_cat'].mean()
    out_df = pd.concat([out_df, pd.DataFrame({f'Proportion {lbl} of Total Spend':s})], axis=1)
    
    ### Too big and not interesting enough :) !
    # # Now loop over regular spending columns
    # for cat in [c for c in pdf.columns if ('COICOP' in c) and (c!=spend_col)]:
    #     s = df.groupby('decile_abs')[cat].mean()
    #     out_df = pd.concat([out_df, pd.DataFrame({f'{dicShort[cat]} spend':s})], axis=1)
    
    s = df.groupby('decile_abs')[spend_col].count()
    out_df = pd.concat([out_df, pd.DataFrame({f'N':s})], axis=1)
        
    out_df = out_df.T
    out_df=out_df.astype(object)
    out_df.loc['N', :] = out_df.loc['N', :].astype(int)
    
    ltx_str = out_df.to_latex(index=True,
                        multicolumn=False,    
                        float_format="{:.2f}".format,  
                        na_rep="--",                       
                        escape=False,                      
                        bold_rows=False,                   
                        header=True)

    with open(latex_out, 'w') as f:
            f.write(ltx_str)
            print('Output to : ', latex_out)  
    
    return out_df

    
def get_decile_demographics():
    from locations import db_pth_uk, db_pth_us
    from locations import pdf_pth_uk, pdf_pth_us, pdf_pth_uk_coffee
    from locations import asset_deb_and_cred_pq_uk, asset_deb_and_cred_pq_us
    from locations import output_folder
    
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import get_labels_and_colour_dics
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dic_COICOP_short, dic_COICOP_colours = get_labels_and_colour_dics(COICOP_labels_df)

    pdf = pd.read_parquet(pdf_pth_us)
    deb_cred_df = pd.read_parquet(asset_deb_and_cred_pq_us)
    df = get_decile_demographic_data(pdf, db_pth_us, dic_COICOP_short, f'{output_folder}/us_decile_demographics.tex', False, deb_cred_df=deb_cred_df)

    pdf = pd.read_parquet(pdf_pth_uk)
    deb_cred_df = pd.read_parquet(asset_deb_and_cred_pq_uk)
    df = get_decile_demographic_data(pdf, db_pth_uk, dic_COICOP_short, f'{output_folder}/uk_decile_demographics.tex', False, deb_cred_df=deb_cred_df)

    pdf = pd.read_parquet(pdf_pth_uk_coffee)
    df = get_decile_demographic_data(pdf, db_pth_uk, dic_COICOP_short, f'{output_folder}/coffee_decile_demographics.tex', True)

def get_spend_ranges(db_pth: str, market: str = 'UK') -> pd.DataFrame:
    """
    Get spend by amount range and COICOP category from DuckDB transaction database.
    We exclude categories based on the COICOP maps lookup (the values in this file 
    should be set to the unidentifiable/ overly generic transactions like).
    Savings and credit card fees are calculated in separate queries
    """
    from A_get_data import get_COICOP_maps, get_code_rules, map_schema, merge_codes
        
    lims = [0, 100, 250, 500, 1000, 2500, 5000, 10000, 10000000]
    lims = [-lims[len(lims)-1-i] for i in range(len(lims)-1)] + lims
    ranges_dict = dict(zip(
        lims,
        ['<-£10K', '(-£10K, -£5K)', '(-£5K, -£2500)', '(-£2500, -£1000)', '(-£1000, -£500)',
         '(-£500, -£250)', '(-£250, -£100)', '(-£100, £0)', '(£0, £100)', '(£100, £250)',
         '(£250, £500)', '(£500, £1000)', '(£1000, £2500)', '(£2500, £5K)', '(£5K, £10K)', '>£10K', 'END']
    ))
    if market.upper() == 'US':
        ranges_dict = {k: v.replace('£', '$') for k, v in ranges_dict.items()}

    dfs = []
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    merge_dic, omit_codes = get_code_rules(COICOP_labels_df)
    omit_string = ', '.join([str(i) for i in omit_codes])
    
    # Schema mappings
    credit_check, sav_check = map_schema(market)

    for i in range(len(lims) - 1):
        mn = lims[i]
        mx = lims[i + 1]

        # Standard COICOP spend
        sql = f"""
            WITH rng AS (SELECT * FROM transactions WHERE amount>{mn} AND amount <{mx}),
                accs AS (SELECT id AS account_id FROM accounts WHERE NOT ({sav_check}))
            SELECT r.COICOP, {mn} AS lower_limit, {mx} AS upper_limit, 
                    COUNT(*) AS cnt, sum(r.amount) AS tot
            FROM rng r INNER JOIN accs a ON r.account_id=a.account_id
            WHERE (NOT r.COICOP IN ({omit_string}))
            AND (NOT r.is_transaction_with_self)
            GROUP BY r.COICOP;"""
        dfs.append(duckdb.connect(db_pth).execute(sql).df())

        # Credit card interest
        sql = f"""
            WITH cc_accounts AS (SELECT id AS account_id FROM accounts WHERE {credit_check}),
                rng AS (SELECT * FROM transactions WHERE amount>{mn} AND amount <{mx} AND brand LIKE '%interest%')
            SELECT 12.21 AS COICOP, {mn} AS lower_limit, {mx} AS upper_limit, 
                    COUNT(*) AS cnt, sum(r.amount) AS tot
            FROM rng r INNER JOIN cc_accounts a ON r.account_id=a.account_id;"""
        dfs.append(duckdb.connect(db_pth).execute(sql).df())

        # Savings
        sql = f"""
            WITH sav_accounts AS (SELECT id AS account_id FROM accounts WHERE {sav_check}),

                non_sav_accounts AS (SELECT id AS account_id FROM accounts WHERE NOT ({sav_check})),

                sav_account_payments AS (SELECT t.user_id, strftime('%Y-%m', t.timestamp) AS month, -t.amount AS amount
                                    FROM transactions t INNER JOIN sav_accounts a 
                                    ON t.account_id=a.account_id
                                    WHERE -t.amount>{mn} AND -t.amount <{mx}),

                non_sav_account_payments AS (SELECT t.user_id, strftime('%Y-%m', t.timestamp) AS month, t.amount
                                    FROM transactions t INNER JOIN non_sav_accounts a 
                                    ON t.account_id=a.account_id
                                    WHERE t.COICOP IN (12.0, 12.20, 12.22)
                                    AND (NOT t.is_transaction_with_self)
                                    AND t.amount>{mn} AND t.amount <{mx}),

                all_payments AS (SELECT user_id, month, amount FROM sav_account_payments
                                 UNION ALL
                                 SELECT user_id, month, amount FROM non_sav_account_payments)   

            SELECT 12.22 AS COICOP, {mn} AS lower_limit, {mx} AS upper_limit, 
                    COUNT(*) AS cnt, sum(amount) AS tot
            FROM all_payments;"""
        dfs.append(duckdb.connect(db_pth).execute(sql).df())

    spend_ranges = pd.concat(dfs).fillna(0).sort_values(['COICOP', 'tot'])
    spend_ranges['range'] = spend_ranges.lower_limit.map(ranges_dict)

    # Now make sure that every COICOP has a full list of ranges (evemn if populated with 0s and NaNs)
    # Sub these in every time 
    ranges = spend_ranges.drop_duplicates(subset='range').sort_values('lower_limit')
    ranges.set_index(ranges.range, inplace=True)
    ranges[['cnt', 'tot']] = 0
    srs=[spend_ranges]
    to_add=0
    for COICOP, rs in spend_ranges.groupby('COICOP'):
        addrows = ranges.drop(index=rs.range.to_list())
        addrows['COICOP'] = COICOP
        srs += [addrows]  
        to_add+=len(addrows)
        
    # Finally, we concatenate together and merge some spend codes into others
    if to_add>0: spend_ranges = pd.concat(srs)
    spend_ranges = merge_codes(spend_ranges, 'COICOP', COICOP_labels_df, 
                            ['lower_limit', 'upper_limit', 'range'],
                            ['cnt', 'tot'])
    spend_ranges = spend_ranges.sort_values(
        ['COICOP', 'lower_limit']).reset_index(drop=True)

    return spend_ranges


def plot_spend_distribution_facet(spend_ranges: pd.DataFrame,
                                   dicShort: dict = dict(),
                                   output_path: str = '',
                                   cols: int = 5,
                                   show_plot: bool = False):
    """
    Plot a faceted bar chart of total spend by category and amount range.

    Parameters:
    - spend_ranges: DataFrame containing columns 'COICOP', 'range', and 'tot'
    - dicShort: Optional dictionary mapping COICOP codes to category names
    - dic_colours: Optional dictionary for hex codes for colours
    - output_path: Optional file path to save the figure (e.g., 'out/cat_plot.png')
    - cols: Number of columns in the subplot grid
    - show_plot: Whether to display the plot with plt.show()
    """
    import math
    ccps = list(spend_ranges.COICOP.unique())
    n = len(ccps)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             constrained_layout=True, sharex=True)
    axes = axes.flatten()

    for idx, ccp in enumerate(ccps):
        ax = axes[idx]
        pldf = spend_ranges[spend_ranges.COICOP == ccp]
        label = dicShort.get(f'COICOP_{ccp}', str(ccp)) if len(dicShort)>0 else str(ccp)      
        
        ax.bar(x=pldf.range, height=pldf.tot, label=label)
        ax.set_title(label)
        ax.set_ylabel('Total detected spend')
        ax.tick_params(axis='x', rotation=90)
        ax.legend()

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    if output_path!='':
        plt.savefig(output_path)
    if show_plot:
        plt.show()
    else:
        plt.close()
        


def output_base_regression_and_robustness_checks():
    from locations import output_folder    
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import combinations, get_labels_and_colour_dics
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dic_COICOP_short, dic_COICOP_colours = get_labels_and_colour_dics(COICOP_labels_df)
    dicShort = {k:v.replace('\n', ' ') for k,v in dic_COICOP_short.items()}
        
    for market in ['UK', 'US']:    
        for lbl, control_variables, net_or_gross in combinations:   
            
            results_pth = f'{output_folder}/{market}_regression_results_{net_or_gross}-control_by_{lbl}.xlsx'
            
            ltx_tbl = pd.read_excel(results_pth, index_col='Unnamed: 0')           
            
            ltx_tbl['label'] = ltx_tbl['Dep. variable code'].map(dicShort)
            ltx_tbl[r'95\% CI'] = ltx_tbl.apply(lambda row: f"[{row['CI Lower']:.2g}, {row['CI Upper']:.2g}]", axis=1)
            
            reg_lbl = f'{net_or_gross} control by {lbl}'
            
            msk = (ltx_tbl.Model == 'Fixed effects') 
            msk = msk & (ltx_tbl.Variable == 'COICOP_14.0')
            ltx_tbl = ltx_tbl[msk].copy() # effects_df[['Category', 'Coef.', 'Std.Err.', 'P>|t|', 'R-squared']].set_index('Category')
            
            N=int(ltx_tbl['N'].iloc[0])
            
            ltx_tbl = ltx_tbl.rename(columns={'label':'Category', 'P>|t|': '$P>|t|$', 'Std.Err.':'Std. Err.'})[['Category', 'Coef.', 'CI Lower', 'CI Upper', 'Std. Err.', '$P>|t|$', 'R-squared']]
            ltx_tbl['Category'] = ltx_tbl['Category'].apply(lambda s: s.replace('&', r'\&')) 
                
            ltx_str = ltx_tbl.to_latex(
                index=False,#                column_format="lcccccc",         
                float_format="{:.2g}".format,  
                na_rep="--",                       # how to render NaNs
                escape=False,                      
                bold_rows=True,                    # make index labels bold (if index=True)
                header=True
            )
            replace_str = r'\bottomrule'
            replace_with_str = r'''\midrule
\multicolumn{6}{r}{$N$} & \multicolumn{1}{r}{replace_this} \\
\bottomrule'''
            replace_with_str = replace_with_str.replace('replace_this', "{:,}".format(N))
            ltx_str = ltx_str.replace(replace_str, replace_with_str)

            ltx_file = results_pth.replace('.xlsx', '.tex')
            with open(ltx_file, 'w') as f:
                f.write(ltx_str)
            print('Read      : ', results_pth)
            print('Output to : ', ltx_file)          


def get_spend_range_plots():
    from locations import db_pth_uk, db_pth_us, output_folder
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import combinations, get_labels_and_colour_dics
    
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dicShort, colour_dic = get_labels_and_colour_dics(COICOP_labels_df)
    
    for market, db_pth in [('UK', db_pth_uk), ('US', db_pth_us)]:
        
        spend_ranges = get_spend_ranges(db_pth, market=market)
        plot_spend_distribution_facet(spend_ranges, dicShort=dicShort, 
                                output_path=f'{output_folder}/category_distribution_facet_{market}.png',
                                cols=5)
   
def get_overall_figures():
    from locations import db_pth_uk, db_pth_us

    for db_pth, market in [(db_pth_uk, 'UK'), (db_pth_us, 'US')]:
        with duckdb.connect(db_pth) as con:
            tot_users = con.execute("""SELECT COUNT(DISTINCT user_id) FROM transactions""").df().iloc[0,0]
            tot_gamblers = con.execute("""SELECT COUNT(DISTINCT user_id) FROM transactions WHERE COICOP=14.0""").df().iloc[0,0]
            trans_count = con.execute("""SELECT COUNT(*) FROM transactions""").df().iloc[0,0]  
            gamb_trans_count = con.execute("""SELECT COUNT(*) FROM transactions WHERE COICOP=14.0""").df().iloc[0,0]  
        
        print(f'####################################################')
        print(f'\nKey headline figures for {market}')
        print(f'### Before asset creation ###')
        print(f'\nTotal unique users \t= {tot_users}    \nTotal unique gamblers \t= {tot_gamblers}')
        print(f'\nTransaction count  \t= {trans_count}  \nGambling transactions \t= {gamb_trans_count}')


def main():
    from locations import pdf_pth_uk_coffee, pdf_pth_uk, output_folder,pdf_pth_us, db_pth_uk
    pdf = pd.read_parquet(pdf_pth_uk)
    pdf_us = pd.read_parquet(pdf_pth_uk)
    pdf_coffee = pd.read_parquet(pdf_pth_uk_coffee)

    get_overall_figures()

    ################################################################
    ## Figures and tables in main article

    # Generate Table ST1 (US) and Table 1 UK
    us_table_1, uk_table_1 = get_overall_demographics_to_latex()

    # Generate Figure 2
    get_figure_2(pdf, pdf_us, pdf_coffee)

    # Generate Figure 3
    get_figure_3()
    
    ################################################################
    ## Figures and tables in Supplementary Information
    
    # Supplementary Figure 1
    plot_classification_proportion(db_pth_uk, f'{output_folder}/cumulative_classification_plot.pdf')
    
    # Supplementary Tables 2, 3 and 4
    get_decile_demographics()
    
    # Regression results to latex
    output_base_regression_and_robustness_checks()
    
    # Supplementary Figures 2 and 3
    get_spend_range_plots()


if __name__ == "__main__":
    main()