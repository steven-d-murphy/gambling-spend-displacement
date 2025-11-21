'''
This file contains everything required to run the regressions to calculate displaced 
gambling spend

Ensure that the file A_get_data.py has been run before running this.
'''

import pandas as pd
from linearmodels.panel import PanelOLS

# Run regressions lots of ways. Controlled by:
#  - spend ['total_spend']
#  - log(net_spend.fillna(0.00)) ['log_spend']
#  - log(1 + abs(total_spend)) ['log_abs_spend']
#  - ... and nothing
# Also for gross gambling and net gambling (as appropriate) 
combinations = [('total_spend', ['total_spend'], 'net'),
                                            ('total_spend', ['total_spend'], 'gross'),
                                            ('log_spend', ['log_spend'], 'gross'),
                                            ('log_abs_spend', ['log_abs_spend'], 'net'),
                                            ('log_abs_spend', ['log_abs_spend'], 'gross'),
                                            ('no_control', [], 'net'),
                                            ('no_control', [], 'gross')]

def run_FERs(df, effect_variable, control_variables=['total_spend']):
    '''
    Run fixed effects regression
    
    df is the asset dataframe outputted from get_pivoted_data()
    
    effect_variable is the independent variable of interest (usually gambling)
    
    control_variables is a list of controls
    '''
    # First, get the dataframe with the user_id and month to the index, 
    # so that we can run the individual fixed effects model
    data = df.copy()
    data['month'] = pd.to_datetime(data.month.astype(str))
    data = data.set_index(['user_id', 'month'])
    
    # What are the spend categories?
    spend_categories = [c for c in df.columns if (('COICOP' in c) and (c!=effect_variable))]
    

    # Make sure everything is a float
    data = data[spend_categories + [effect_variable] + 
                control_variables].astype(float)

    FER_results = {}
    FER_results_quintiles = {}
    vif = pd.DataFrame()
    for category in spend_categories:        
        # Define the dependent variable. Same for both models
        y = data[category]

        # Get result without quintiles dummies
        print(f'Running regressions to predict {category} from {effect_variable} (no dummies)')
        X = data[[effect_variable] + control_variables]
        
        
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        results = model.fit(cov_type='clustered', cluster_entity=True) # \exists within entity corellation
        FER_results[category] = results

    print('Done')

    return FER_results, FER_results_quintiles

# Function to extract model info from linearmodels
def extract_linearmodels_info(model):
    coef = model.params
    std_err = model.std_errors
    p_values = model.pvalues
    conf_int = model.conf_int()
    r_squared = model.rsquared
    n_obs = model.nobs
    
    data = {
        'Variable': coef.index,
        'Coef.': coef.values,
        'Std.Err.': std_err.values,
        'P>|t|': p_values.values,
        'CI Lower': conf_int.iloc[:, 0].values,
        'CI Upper': conf_int.iloc[:, 1].values,
        'R-squared': [r_squared] * len(coef),
        'N': [n_obs] * len(coef)
    }
    df = pd.DataFrame(data)
    return df

def get_tables_of_results(F_results, F_quint_results, COICOP_labels_df, xlsx_out):
    lst = []
    ldf = pd.DataFrame()
    for c in F_results.keys():
        FER_model = F_results[c]

        FER_res = extract_linearmodels_info(FER_model)

        # First get long version
        FER_res['Model'] = 'Fixed effects'

        FER_res['Dep. variable code'] = c
        ldf = pd.concat([ldf, FER_res])

    ldf.loc[(ldf['Dep. variable code'] == 'unidentified'), 'Dep. variable code'] = 'COICOP_00'
    ldf = ldf.reset_index(drop=True)
    ldf.to_excel(xlsx_out)

    return ldf

def get_labels_and_colour_dics(lkp_df, colour_scheme='ryb'):
    '''
    Takes the lookup dataframe and returns dictionaries of labels for the plots      
    '''
    # dictionary of long COICOP labels
    dic_COICOP_long = pd.Series(lkp_df.long_labels.to_list(),  index=lkp_df.COICOP.to_list()).to_dict()
    # dictionary of short COICOP labels
    dic_COICOP_short = pd.Series(lkp_df.short_labels.to_list(),  index=lkp_df.COICOP.to_list()).to_dict()
    # dictionary of colours for COICOP labels
    if (colour_scheme == 'ryb'):
        dic_COICOP_colours =  pd.Series(lkp_df.nature_rybp.to_list(),  index=lkp_df.COICOP.to_list()).to_dict()
    elif (colour_scheme == 'original'):
        dic_COICOP_colours =  pd.Series(lkp_df.colour.to_list(),  index=lkp_df.COICOP.to_list()).to_dict()
    elif (colour_scheme == 'new'):
        dic_COICOP_colours = {}
    else: # Set to ryb by default
        dic_COICOP_colours =  pd.Series(lkp_df.nature_rybp.to_list(),  index=lkp_df.COICOP.to_list()).to_dict()

    dic_COICOP_colours['Combined_COICOP'] = '#D3D3D3'
             
    return dic_COICOP_long, dic_COICOP_short, dic_COICOP_colours

def main():

    from locations import output_folder
    from A_get_data import  get_COICOP_maps
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    dic, dicShort, colour_dic = get_labels_and_colour_dics(COICOP_labels_df)

    # Run regressions for both US and UK data
    for market in ['US', 'UK']:
        
        if market == 'UK':
            from locations import asset_pq_uk, pdf_pth_uk, asset_pq_gross_uk, pdf_pth_gross_uk, asset_deb_and_cred_pq_uk, asset_deb_cred_net_pq_uk
            adf_net = pd.read_parquet(asset_pq_uk)
            pdf_net = pd.read_parquet(pdf_pth_uk)

            adf_gross = pd.read_parquet(asset_pq_gross_uk)
            pdf_gross = pd.read_parquet(pdf_pth_gross_uk)

            adf_deb_and_cred = pd.read_parquet(asset_deb_and_cred_pq_uk)
            deb_cred_net = pd.read_parquet(asset_deb_cred_net_pq_uk)

        
        else:
            from locations import asset_pq_us, pdf_pth_us, asset_pq_gross_us, pdf_pth_gross_us, asset_deb_and_cred_pq_us, asset_deb_cred_net_pq_us
            adf_net = pd.read_parquet(asset_pq_us)
            pdf_net = pd.read_parquet(pdf_pth_us)
            adf_gross = pd.read_parquet(asset_pq_gross_us)
            pdf_gross = pd.read_parquet(pdf_pth_gross_us)
        
        dfs = []
        for lbl, control_variables, net_or_gross in combinations:
            reg_lbl = f'{net_or_gross} control by {lbl}'
            
            if 'net'==net_or_gross:
                use_pdf = pdf_net
            else: use_pdf = pdf_gross
            
            # Now we run regressions - first net:
            print(f'\nRunning regressions on {market} data with gambling variable as {net_or_gross} debits, controlled by {control_variables}')
            FER_results_net, FER_results_quintiles_net = run_FERs(
                use_pdf, 
                'COICOP_14.0', 
                control_variables=control_variables) 
            ldf = get_tables_of_results(FER_results_net, 
                                        FER_results_quintiles_net, 
                                        COICOP_dic, 
                                        f'{output_folder}/{market}_regression_results_{net_or_gross}-control_by_{lbl}.xlsx')
            
            ldf['version']= reg_lbl
            dfs += [ldf.copy()]
            
        ldf = pd.concat(dfs)
        ldf['label'] = ldf['Dep. variable code'].map(dicShort)
        ldf.to_excel(f'{output_folder}/{market}_regression_results_combined.xlsx')
        
    
if __name__ == "__main__":
    main()