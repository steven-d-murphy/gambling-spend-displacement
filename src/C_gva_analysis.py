'''
This file contains everything required to analyse the GVA impact of displaced spend
to gambling from other categories

Ensure that the file "B_regression_analysis.py" has been run before running this.
'''


import pandas as pd
import numpy as np
from scipy.stats import norm

def prepare_effects_df(ldf: pd.DataFrame, get_maps_fn, get_labels_fn) -> pd.DataFrame:
    # Load COICOP labels and colour mappings
    COICOP_labels_df, COICOP_dic = get_maps_fn()
    dic, dicShort, colour_dic = get_labels_fn(COICOP_labels_df)

    # Filter to the relevant regression results and add readable category names
    df = ldf[(ldf.Model == 'Fixed effects') & (ldf.Variable == 'COICOP_14.0')].copy()
    df['Category'] = df['Dep. variable code'].map(dicShort)
    return df

def calculate_L1_effects(adf: pd.DataFrame) -> pd.DataFrame:
    # Get gva tables
    from locations import gva_tables_xlsx
    gva_df = pd.read_excel(gva_tables_xlsx, sheet_name='CPA_COICOP_map')

    # Get the weighted sum of all GVA effects. 
    # This is effectively the expected GVA effect of a pound spent by a household   
    weighted_sum_of_all_effects = (
        (gva_df['Total domestic use (CPA)'] / gva_df['Total domestic use (CPA)'].sum()) 
        * gva_df['Effect (CPA)']
    ).sum()

    # Remove gambling to lotteries row. No reweighting required as CPA 95 is not split and we intend to 
    # use the same effect size for both gambling and lotteries
    gambling_effect = gva_df.loc[gva_df['CPA'] == 92, 'Effect (CPA)'].iloc[0]
    gva_df.loc[gva_df['CPA'] == 92, 'my_coicop'] = 15

    # Get the effects for insurance and credit cards, then remove COICOP 12 - we deal with insurance and financial services differently
    insurance_effect = gva_df[
        (gva_df['my_coicop'] == '[reallocate in python]') & (gva_df['CPA'] == '65.1-2')
    ]['Effect (CPA)'].iloc[0]

    credit_cards_effect = gva_df[
        (gva_df['my_coicop'] == '[reallocate in python]') & (gva_df['CPA'] == 64)
    ]['Effect (CPA)'].iloc[0]

    gva_df = gva_df[gva_df.my_coicop != '[reallocate in python]']

    # Weight the CPA effects to the level 3 COICOP
    gva_df['CPA_weight_times_effect'] = gva_df['CPA Weight in COICOP'] * gva_df['Effect (CPA)']

    # What is the total domestic use of each CPA allocated to each COICOP?
    gva_df['CPA_total_use_in_COICOP'] = gva_df['Proportion of CPA in each COICOP'] * gva_df['Total domestic use (CPA)']

    # How do we need to weight the level 3 COICOP values when we aggregate up to level 1?
    l3_weights = (
        gva_df.groupby(['my_coicop', 'COICOP'])['CPA_total_use_in_COICOP']
        .sum()
        .reset_index(name='L3_total_use')
    )
    l3_weights['L1_total_use'] = l3_weights.groupby('my_coicop')['L3_total_use'].transform('sum')
    l3_weights['L3_to_L1_weight'] = l3_weights['L3_total_use'] / l3_weights['L1_total_use']

    # What are the GVA effects for each level 3 COICOP?
    l3_effects = (
        gva_df.groupby(['my_coicop', 'COICOP'])['CPA_weight_times_effect']
        .sum()
        .reset_index(name='COICOP_effect')
    )

    # What is the GVA effect for each level 1 COICOP?
    l3_effects = pd.merge(l3_effects, l3_weights[['L3_to_L1_weight', 'COICOP']], on='COICOP')
    l3_effects['COICOP_weight_times_effect'] = l3_effects['COICOP_effect'] * l3_effects['L3_to_L1_weight']
    L1_effects = l3_effects.groupby('my_coicop')['COICOP_weight_times_effect'].sum().reset_index(name='effect')

    # Add previously extracted categories
    L1_effects.loc[len(L1_effects.index)] = [12.1, insurance_effect]
    L1_effects.loc[len(L1_effects.index)] = [12.21, credit_cards_effect]
    L1_effects.loc[len(L1_effects.index)] = [12.25, weighted_sum_of_all_effects] # Loan repayments - assume money has already been spent at regular distribution
    L1_effects.loc[len(L1_effects.index)] = [14.0, gambling_effect]
    L1_effects.loc[len(L1_effects.index)] = [13.1, weighted_sum_of_all_effects] # general purpose retailers
    L1_effects.loc[len(L1_effects.index)] = [13.2, 0.0]                         # Direct taxation excluded from GVA
    L1_effects.loc[len(L1_effects.index)] = [13.3, weighted_sum_of_all_effects] # Polit. donations and charitible. Charitible>>political in GB. 
                                                                                # Assume charitible donations result in sgvadary spending at everage effect

    # Get sum of effects, weighted by the spending rates we observe in our panel.
    # This will be the effect allocated to savings, investments and pensions
    total_spend_by_category = -adf.groupby('COICOP')['sum_of_spends'].sum()
    L1_effects = pd.merge(L1_effects, total_spend_by_category, left_on='my_coicop', right_index=True, how='outer')
    L1_effects = L1_effects.sort_values('my_coicop').set_index('my_coicop')

    calculated = L1_effects[L1_effects.effect.notna()]
    non_savings_weighted = (calculated.effect * (calculated.sum_of_spends / calculated.sum_of_spends.sum())).sum()

    for ccp in [12.22, 12.23, 12.24]:
        L1_effects.loc[ccp, 'effect'] = non_savings_weighted

    return L1_effects.reset_index()

def merge_effects_with_regression(effects_df: pd.DataFrame, L1_effects: pd.DataFrame) -> pd.DataFrame:
    # Create a mapping from float COICOP codes to their string format used in the regression results
    ccp_map = dict(zip(
        effects_df['Dep. variable code'].apply(lambda s: float(s.split('_')[1])),
        effects_df['Dep. variable code']
    ))
    ccp_map[14.0] = 'COICOP_14.0'
    L1_effects['Dep. variable code'] = L1_effects['my_coicop'].map(ccp_map)

    # Merge calculated GVA effects with regression coefficients
    return pd.merge(effects_df, L1_effects, on='Dep. variable code', how='inner')

def estimate_displaced_GVA(eff_df, gambling_effect, gambling_size):
    effects_df = eff_df.copy().set_index('my_coicop')
    
    # Apply the regression coefficients to the calculated GVA effects
    s = effects_df['Coef.'] / effects_df['Coef.'].sum() # normalise coefficients - assume sum(displacement) = total gambling
    displaced_effect_by_category = s * effects_df.effect   # Apply coefficients to the estimated effects
    displaced_effect = displaced_effect_by_category.sum()  # Total displaced effect

    # Calculate gross value added (GVA)
    gamb_GVA = gambling_effect * gambling_size
    displaced_GVA = displaced_effect * gambling_size
    net_benefit = gamb_GVA - displaced_GVA

    return displaced_effect, gamb_GVA, displaced_GVA, net_benefit

def calculate_CIs(effects_df, pdf, gambling_effect, gamb_GVA, gambling_size, confidence_level = 0.95):
    # As we estimated the panel regressions in each category separately, we do not have a 
    # measure of correlation between each category coefficient
    # Instead, we use correlation between categories in the base data  
    corrmat = pdf[[c for c in pdf.columns if 'OICOP' in c and '14' not in c]].corr()

    # Use the standard errors in the estimated coefficients
    SEs = effects_df.set_index('Dep. variable code')['Std.Err.']

    # Critical value for confidence interval
    alpha = 1 - confidence_level
    crit_val = norm.ppf(1 - alpha / 2)

    # Now calculate the variance and SD in the summed estimate
    var_displaced = (SEs**2).sum()
    for i in corrmat.columns:
        for j in corrmat.columns:
            if j > i:
                var_displaced += 2 * corrmat.loc[i, j] * SEs.loc[i] * SEs.loc[j]

    # Estimate is scaled, so scale the standard deviation too
    SD_displaced_scaled = np.sqrt(var_displaced) / abs(effects_df['Coef.'].sum())
    mean_displaced = (effects_df['Coef.'] / effects_df['Coef.'].sum() * effects_df.effect).sum()

    # Confidence intervals for displaced effect and GVA
    displaced_CIs = [mean_displaced - crit_val * SD_displaced_scaled,
                     mean_displaced + crit_val * SD_displaced_scaled]

    displaced_GVA_CIs = [gambling_size * val for val in displaced_CIs]
    GVA_benefit_CIs = [f'{(gamb_GVA - val) / 1e6:.2f}' for val in displaced_GVA_CIs]

    return displaced_CIs, displaced_GVA_CIs, GVA_benefit_CIs

def print_gva_results_to_screen(gamb_industry_size, 
                                 gambling_effect,
                                 displaced_effect, 
                                 gamb_GVA, 
                                 displaced_GVA, 
                                 displaced_CIs,
                                 displaced_GVA_CIs):
    """Prints easily readable results from the GVA analysis"""
    ind = []
    vals = []
    
    print('\n')
    print(f"Gambling effect           = {gambling_effect:.2f}")
    ind += [f"Gambling effect"]
    vals += [f"{gambling_effect:.2f}"]

    print(f"Displaced spending effect = {displaced_effect:.2f} [{displaced_CIs[0]:.2f}, {displaced_CIs[1]:.2f}]")
    
    ind += [f"Displaced spending effect"]
    vals += [f"{displaced_effect:.2f} [{displaced_CIs[0]:.2f}, {displaced_CIs[1]:.2f}]"]

    print(f"""GVA benefit of increased gambling activity is {100*(gambling_effect - displaced_effect):.1f} percentage points [{
            100*(gambling_effect - displaced_CIs[1]):.1f}, {100*(gambling_effect - displaced_CIs[0]):.1f}]""")
        
    ind += [f"""GVA benefit of increased gambling activity"""]
    vals += [f"""{100*(gambling_effect - displaced_effect):.1f} percentage points [{100*(gambling_effect - displaced_CIs[1]):.1f}, {100*(gambling_effect - displaced_CIs[0]):.1f}]"""]

    print(f"GB gambling industry size [2023/24] = £{(gamb_industry_size/1000000):.2f} million")

    ind += [f"""GB gambling industry size [2023/24]"""]
    vals += [f"""£{(gamb_industry_size/1000000):.2f} million"""]

    print(f"Gambling industry GVA     = £{(gamb_GVA/1000000):.2f} million")
    
    ind += [f"""Gambling industry GVA"""]
    vals += [f"""£{(gamb_GVA/1000000):.2f} million"""]
    
    print(f"""Displaced spending GVA    = £{(displaced_GVA/1000000):.2f} million [£{
        (displaced_GVA_CIs[0]/1000000):.2f}m, £{(displaced_GVA_CIs[1]/1000000):.2f}m]""")
    
    ind += [f"""Displaced spending GVA"""]
    vals += [f"""£{(displaced_GVA/1000000):.2f} million [£{
        (displaced_GVA_CIs[0]/1000000):.2f}m, £{(displaced_GVA_CIs[1]/1000000):.2f}m]"""]

    print(f"""Gambling industry net GVA benefit = £{((gamb_GVA - displaced_GVA)/1000000):.2f} million [£{
        ((gamb_GVA - displaced_GVA_CIs[1])/1000000):.2f}m, £{((gamb_GVA - displaced_GVA_CIs[0])/1000000):.2f}m]""")
    
    ind += [f"""Gambling industry net GVA benefit"""]
    vals += [f"""£{((gamb_GVA - displaced_GVA)/1000000):.2f} million [£{
        ((gamb_GVA - displaced_GVA_CIs[1])/1000000):.2f}m, £{((gamb_GVA - displaced_GVA_CIs[0])/1000000):.2f}m]"""]
    
    out_df = pd.DataFrame({'value':vals, 'measure':ind}).set_index('measure', drop=True)
    
    return out_df
    
def main():
    
    from locations import pdf_pth_uk, asset_pq_uk, asset_pq_gross_uk, gva_tables_xlsx, output_folder
    from A_get_data import get_COICOP_maps
    from B_regression_analysis import get_labels_and_colour_dics

    adf = pd.read_parquet(asset_pq_uk)
    pdf = pd.read_parquet(pdf_pth_uk)
    adf_gross = pd.read_parquet(asset_pq_gross_uk)

    gamb_industry_size=11545610000

    ldf = pd.read_excel(f'{output_folder}/UK_regression_results_combined.xlsx', index_col='Unnamed: 0')
    ldf_net = ldf[ldf['version']=='net control by total_spend']
    ldf_gross = ldf[ldf['version']=='gross control by total_spend']

    effects_df = prepare_effects_df(ldf_net, get_COICOP_maps, get_labels_and_colour_dics)

    L1_effects = calculate_L1_effects(adf)

    gambling_effect = L1_effects.set_index('my_coicop').loc[14.0].effect

    effects_df = merge_effects_with_regression(effects_df, L1_effects)

    displaced_effect, gamb_GVA, displaced_GVA, net_benefit = estimate_displaced_GVA(
        effects_df, gambling_effect=gambling_effect, gambling_size=gamb_industry_size
    )

    displaced_CIs, displaced_GVA_CIs, gva_benefit_CIs = calculate_CIs(
        effects_df, pdf, gambling_effect=gambling_effect, 
        gamb_GVA=gamb_GVA, gambling_size=gamb_industry_size
    )
    print('Net spend effects:')
    gva_net = print_gva_results_to_screen(gamb_industry_size, 
                                gambling_effect, 
                                displaced_effect, 
                                gamb_GVA, 
                                displaced_GVA, 
                                displaced_CIs, 
                                displaced_GVA_CIs)

    effects_df = prepare_effects_df(ldf_gross, get_COICOP_maps, get_labels_and_colour_dics)

    L1_effects = calculate_L1_effects(adf_gross)

    gambling_effect = L1_effects.set_index('my_coicop').loc[14.0].effect

    effects_df = merge_effects_with_regression(effects_df, L1_effects)

    displaced_effect, gamb_GVA, displaced_GVA, net_benefit = estimate_displaced_GVA(
        effects_df, gambling_effect=gambling_effect, gambling_size=gamb_industry_size
    )

    displaced_CIs, displaced_GVA_CIs, gva_benefit_CIs = calculate_CIs(
        effects_df, pdf, gambling_effect=gambling_effect, 
        gamb_GVA=gamb_GVA, gambling_size=gamb_industry_size
    )
    print('Gross spend effects:')
    gva_gross = print_gva_results_to_screen(gamb_industry_size, 
                                gambling_effect, 
                                displaced_effect, 
                                gamb_GVA, 
                                displaced_GVA, 
                                displaced_CIs, 
                                displaced_GVA_CIs)

    gva_joined = pd.concat([gva_net, gva_gross], axis=1)
    gva_joined.columns = ['Net gambling spend', 'Gross gambling spend']
    gva_joined.to_excel(f'{output_folder}/gva_results.xlsx', index=True)
    
    print(gva_joined)
        
if __name__ == "__main__":
    main()