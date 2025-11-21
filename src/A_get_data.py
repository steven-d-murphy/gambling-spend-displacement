'''
This file contains everything required to build the data assets used for the gambling displacement analysis

Ensure that the file "locations.py" is updated before running this.
'''
import duckdb, pandas as pd, numpy as np
from locations import (lkp_xlsx, 
                       spend_sql,
                       savings_sql, 
                       credit_card_sql, 
                       coffee_sql)
from locations import (db_pth_uk,
                       asset_pq_uk, 
                       pdf_pth_uk, 
                       asset_pq_gross_uk, 
                       pdf_pth_gross_uk, 
                       asset_deb_and_cred_pq_uk, 
                       asset_deb_cred_net_pq_uk,
                       asset_pq_uk_coffee,
                       pdf_pth_uk_coffee,
                       pop_asset_pq_uk, 
                       pop_pdf_pth_uk)
from locations import (db_pth_us,
                       asset_pq_us, 
                       pdf_pth_us, 
                       asset_pq_gross_us, 
                       pdf_pth_gross_us, 
                       asset_deb_and_cred_pq_us, 
                       asset_deb_cred_net_pq_us,
                       pop_asset_pq_us, 
                       pop_pdf_pth_us)

def get_COICOP_maps():
    """
    Get classification maps - from our classification system, which is a 
    modified and extended version of the COICOP system
    """
    COICOP_labels_df = pd.read_excel(lkp_xlsx, sheet_name='lookup', index_col='code')
    COICOP_dic = COICOP_labels_df.label.to_dict()
    return COICOP_labels_df, COICOP_dic

def get_code_rules(COICOP_labels_df):
    """
    Some classification codes get folded into others
    Some are ommitted
    Return 1, merge_dic : A dictionary of codes, key, that get merged in to other codes,
    value
    Return 2, omit_codes : A list of codes that are to be omitted.
    Note: Apply merge_dic before omit_codes
    """
    merge_dic = COICOP_labels_df.fold_into_code.to_dict()
    merge_dic = {k:v for k,v in merge_dic.items() if (k!=v) and (v!='omit')}
    omit_codes = list(COICOP_labels_df[COICOP_labels_df.fold_into_code=='omit'].index)
    
    for k,v in merge_dic.items():
        if v in omit_codes: omit_codes += [k]
    
    return merge_dic, omit_codes

def map_schema(market):
    """Different databases have slightly different schema. Get the correct filter condition"""    
    if market.lower() == 'uk': # This is latest uk dataset
        credit_check = "type = 'CREDIT'"
        sav_check = "type LIKE '%SAV%'"
    elif market.lower() == 'us':
        credit_check = "type = 'credit'"
        sav_check = "account_subtype = 'savings'"
    else: # Historic uk dataset
        credit_check = "type = 'card'"
        sav_check = "account_type LIKE '%SAV%'"
    return credit_check, sav_check

def merge_codes(df_in, lbl, COICOP_labels_df, group_cols, sum_cols):
    """
    This is for merging classified records into each other.
    lbl will be a column (probably COICOP or code)
    
    COICOP_labels_df is the dataframe from the control file. This contains information on
    what is to be merged or omitted.
    For example. We might want to merge COICOP 13.1 into COICOP 13.0, by adding the 
    values in the sum_cols columns for 13.1 into those for 13.0. 
    
    group_cols are the columns over which we group (e.g., user_id, month)
    sum_cols are what we sum over
    Assume any further columns are 'secondary' labels, that are 1:1 with the label column 
    """
    df = df_in.copy()
    cols = df.columns # Order of columns for output
    
    # Get merge dictionary
    d, omit_codes = get_code_rules(COICOP_labels_df)    
    
    # Are there additional labels that go 1:1 with the lbl column?
    lbl_cols = [c for c in cols if not (c in ([lbl] + group_cols + sum_cols))]
    join_lookup = df[[lbl] + lbl_cols].drop_duplicates(subset=lbl, keep='first')
    
    dfs = []
    for v in set(d.values()): # for every target label
        ks_and_v =  set([v] + [ke for ke,va in d.items() if va==v]) # all labels that will end up as v
        msk = df[lbl].isin(ks_and_v)
        if sum(msk) > 0:    
            mergers = df[msk]
            df = df[~msk]
            merged = mergers.groupby(group_cols)[sum_cols].sum().reset_index()
            merged[lbl] = v
            dfs += [merged]
            
    new_rows = pd.concat(dfs, axis=0)
    
    # Are there any extra lables associated with each unique lbl? (e.g., COICOP label associated with code)    
    if len(lbl_cols)>0:
        new_rows = pd.merge(new_rows, 
                            join_lookup,
                            on=lbl, 
                            how='left')        
        
    new_rows = new_rows[cols] # Sort column order and concat for output
    df = pd.concat([df, new_rows], axis=0).sort_values(lbl).reset_index(drop=True)
    return df

def get_asset(db_pth, COICOP_labels_df, is_net = True,  market: str = 'uk', drop_non_gamblers: bool=True):
    '''
    Gets data asset from SQL and do initial preprocessing
    '''
    from locations import spend_sql, savings_sql, credit_card_sql

    # Schema mappings
    credit_check, sav_check = map_schema(market)

    with duckdb.connect(db_pth) as con:
        query = open(spend_sql, "r").read().replace('sav_check', sav_check).replace('credit_check', credit_check)
        adf_deb_and_cred = con.execute(query).df()
        adf = adf_deb_and_cred.copy()

        if is_net:
            net_adf = adf[adf.COICOP.isin([14.0, 15.0])].groupby(
                ['user_id', 'month', 'COICOP']
                ).agg({'count':'sum', 'sum':'sum'}).reset_index()
            adf = pd.concat([ adf[~adf.COICOP.isin([14.0, 15.0]) & adf.is_debit].drop(columns='is_debit'), net_adf])
        else:
            adf=adf[adf.is_debit].drop(columns='is_debit')

        query = open(savings_sql, "r").read().replace('sav_check', sav_check).replace('credit_check', credit_check)
        sdf = con.execute(query).df()

        query = open(credit_card_sql, "r").read().replace('sav_check', sav_check).replace('credit_check', credit_check)
        cdf = con.execute(query).df()

    # Ensure that we only have relevant months in the savings and credit card data
    sdf = pd.merge(sdf, adf[['user_id', 'month']].drop_duplicates(), on=['user_id','month'], how='inner').rename(columns={'sav_count':'count', 'sav_sum':'sum'})
    cdf = pd.merge(cdf, adf[['user_id', 'month']].drop_duplicates(), on=['user_id','month'], how='inner')

    # Append credit card interest and savings totals
    adf = pd.concat([adf, cdf, sdf])

    # Drop non-gamblers
    if drop_non_gamblers:
        uids = adf[adf.COICOP==14.0].user_id.drop_duplicates().to_list()
        adf = adf[adf.user_id.isin(uids)]
        adf_deb_and_cred = adf_deb_and_cred[adf_deb_and_cred.user_id.isin(uids)]

    # Get COICOP labels
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    grp = ['user_id', 'month', 'COICOP']
    adf.columns = grp + ['count_of_spends', 'sum_of_spends']
    adf['COICOP_label'] = adf.COICOP.map(COICOP_dic)
    adf.sort_values('COICOP',inplace=True) # Ensure that the COICOP values come out in the right order
    adf['COICOP_str'] = 'COICOP_' + (adf.COICOP.apply(str))

    # Merge categories that can go together.
    adf = merge_codes(adf, "COICOP", COICOP_labels_df, 
                      ['user_id', 'month'], 
                      ['count_of_spends', 'sum_of_spends'])
    adf_deb_and_cred = merge_codes(adf_deb_and_cred, "COICOP", COICOP_labels_df, 
                      ['user_id', 'month', 'is_debit'], 
                      ['count', 'sum'])
    
    # Now omit codes that we want to omit
    d, omit_codes = get_code_rules(COICOP_labels_df)  
    adf = adf[~adf["COICOP"].isin(omit_codes)]
    adf_deb_and_cred = adf_deb_and_cred[~adf_deb_and_cred["COICOP"].isin(omit_codes)]

    # Finally kill income categories that are debits (likely these
    # result from misclassification)
    msk = (adf.COICOP<0.0) & (adf.sum_of_spends<0.0)
    adf = adf[~msk]

    msk = (adf_deb_and_cred.COICOP<0.0) & (adf_deb_and_cred['sum']<0.0)
    adf_deb_and_cred = adf_deb_and_cred[~msk]    

    return adf, adf_deb_and_cred


def add_percentile_decile_and_quintile(df_in, sort_col, suff):
    '''
    Gets percentile of gambling activity within month
    sort_col could be e.g., gambling or prop_gambled
    '''

    pcol = 'percentile_' + suff
    dcol = 'decile_' + suff
    qcol = 'quintile_' + suff

    df = df_in.sort_values(sort_col, ascending=True).copy()

    # Group by month and rank users within each month based on their gambling value
    for month, group in df.groupby('month'):
        non_zero_users = group[group[sort_col] > 0]
        ranked_users = non_zero_users.sort_values(sort_col, ascending=True)
        rank = ranked_users.reset_index().index.to_list()
        ranked_users[pcol]  = [1 + int(100*i/len(rank)) for i in rank]
        df.loc[df['month'] == month, pcol] = ranked_users[pcol]

    # Also get the quintile and decile
    df[pcol] = df[pcol].fillna(0).apply(int)
    df[dcol] = (df[pcol] + 9) // 10
    df[qcol] = (df[pcol] + 19) // 20 

    # If there are no gambling deposit at all, have a 0
    df.loc[df[sort_col] == 0, pcol] = 0
    df.loc[df[sort_col] == 0, dcol] = 0
    df.loc[df[sort_col] == 0, qcol] = 0
        
    return df

def get_pivoted_data(adf, 
                     COICOP_str,                # The label of the primary variable (COICOP_14.0 or COICOP_16.0)
                     lbl,                       # The label of the primary variable (gambling or coffee)
                     quantiles=pd.DataFrame(),  # Dataframe with user_id, month, [quantile colums]. Join to this, otherwise...
                     top_trim_quantile=0.9975): # Proportion of user_id+months to remove at the top end, then calculate quantiles
    '''
    Takes:
    - adf, the data asset (output from get_asset()) 
    - COICOP_str, the label of the primary variable (COICOP_14.0 or COICOP_16.0)
    - lbl, The label of the primary variable (gambling or coffee)
    - quantiles=pd.DataFrame(),  Dataframe with user_id, month, [quantile colums]. Join to this, otherwise...
    - top_trim_quantile=0.9975, Proportion of user_id+months to remove at the top end, then calculate quantiles
    
    and outputs 
    A pivoted dataset of spending in each person-month by category

    Includes 
        calculating monthly spend
        either:
            getting the quantiles
        or:
            Joining to dataframe of previously calculated quantiles (for using quantiles based on debits, with net spend dataset)
    '''

    idx_cols = ['user_id', 'month']
    pdf = pd.pivot_table(adf,
                     index=idx_cols,
                     columns='COICOP_str',
                     values='sum_of_spends').fillna(0)
       

    # Reverse the values so debits are positive. Ensure correct column order
    COICOP_cols = ['COICOP_' + str(v) for v in adf.COICOP.drop_duplicates().sort_values().to_list()]
    pdf = -pdf[COICOP_cols]

    # Get totals
    monthly_totals = adf.groupby(['user_id', 
                                  'month']).sum_of_spends.sum().reset_index().rename(
                                      columns={'sum_of_spends':'total_spend'})
    pdf = pd.merge(pdf, monthly_totals, how='left', on=idx_cols)
    pdf['total_spend'] = -pdf['total_spend'].fillna(0) # reverse sign and fill nas

    # Proportion gambled - spend   
    pdf['log_spend'] = (pdf['total_spend']).apply(np.log10).fillna(0.0)
    pdf['log_abs_spend'] = (pdf['total_spend'].abs() + 1).apply(np.log10)
    pdf[f'proportion_debits_{lbl}'] = pdf[COICOP_str] / pdf['total_spend']

    if len(quantiles)>0:
        pdf = pd.merge(pdf, quantiles, how='right', on=idx_cols)
    else:
        # Kill top top_trim_quantile percent of records
        lim = pdf.total_spend.quantile(top_trim_quantile)
        pdf = pdf[pdf.total_spend < lim]

        # Keep records with month spend over £100/$100
        pdf = pdf[pdf.total_spend > 100.0]

        # Only retain months when we have at least 10 gamblers
        mcs = pdf[pdf[COICOP_str] != 0.0].groupby('month').user_id.nunique() 
        over_10_uids = mcs.loc[(mcs >= 10)].index.to_list()
        pdf = pdf[pdf.month.isin(over_10_uids)]

        # This may have created new 'non-gamblers' from the perspective of the revised time period.
        # Drop users with no gambling months
        gamblers = pdf[pdf[COICOP_str] != 0.0].user_id.drop_duplicates().to_list()
        pdf = pdf[pdf.user_id.isin(gamblers)]

        # Get ntiles and log spend
        pdf = add_percentile_decile_and_quintile(pdf, COICOP_str, 'abs')
    
    return pdf

def merge_into_adf(adf_in, sub_values, COICOP, COICOP_label, COICOP_str):
    '''Pull a sub category out of another COICOP in the long "adf" dataframe 
    E.g., Pull coffee shops out of COICOP 1
    '''

    # Get relevent user_id+month values
    inx = ['user_id', 'month']
    keep_indices = pd.merge(adf_in[inx].drop_duplicates(), 
                       sub_values[inx].drop_duplicates(), 
                       how='inner', 
                       on=inx).set_index(inx).index.to_list()
    sub_values = sub_values.set_index(inx).loc[keep_indices].reset_index(drop=False)

    # Remove values from existing COICOP
    adf_new = pd.merge(adf_in, 
                       sub_values[['user_id', 'month', 'COICOP', 'count', 'sum']], 
                       on=['user_id', 'month', 'COICOP'], how='left').fillna(0)
    adf_new['count_of_spends'] -= adf_new['count'].astype(int)
    adf_new['sum_of_spends'] -= adf_new['sum']
    adf_new = adf_new.drop(columns=['count', 'sum'])

    # Add new values in
    sub_values['COICOP'] = COICOP
    sub_values['COICOP_label'] = COICOP_label
    sub_values['COICOP_str'] = COICOP_str
    adf_new = pd.concat([sub_values.rename(columns={'count':'count_of_spends', 
                                                       'sum':'sum_of_spends'}), adf_new])

    return adf_new

def pull_coffee_into_new_category(db_pth, adf, market):
    '''Spend at coffee shops is by default in COICOP 1
    This calculates coffee shop spend from the database and 
    extracts this from the long form database into a new category'''

    # Schema mappings
    credit_check, sav_check = map_schema(market)

    with duckdb.connect(db_pth) as con:
        query = open(coffee_sql, "r").read().replace('sav_check', sav_check).replace('credit_check', credit_check)
        coffee_spends = con.execute(query).df()

    adf_coffee = merge_into_adf(adf, coffee_spends, 16.0, 'Coffee shops', 'COICOP_16.0')
    return adf_coffee

def get_debits_credits_and_net_by_decile(adf_deb_and_cred,
                                         pdf,
                                         decile_col = 'decile_abs'):
    """
    Construct dataframe with debits, credits and net for all categories. Requires the second output from get_asset()
    and a pivoted database from get_pivoted_data() for the decile column
    
    Only includes user_id+month combinations included in pdf
    """   
    
    debit_vals = pd.merge(pdf[['user_id', 'month', decile_col]], 
                        adf_deb_and_cred[adf_deb_and_cred.is_debit], 
                        on=['user_id', 'month'],
                        how='outer')
    debit_vals = debit_vals[debit_vals[decile_col].notna() & debit_vals['COICOP'].notna()]
    
    credit_vals = pd.merge(pdf[['user_id', 'month', decile_col]], 
                        adf_deb_and_cred[~(adf_deb_and_cred.is_debit)], 
                        on=['user_id', 'month'],
                        how='outer')
    credit_vals = credit_vals[credit_vals[decile_col].notna() & credit_vals['COICOP'].notna()]

    deb_cred_net = pd.merge(debit_vals, credit_vals, 
                            on=['user_id','month', decile_col,'COICOP'], 
                            how='outer').drop(
                                columns=['is_debit_x', 'is_debit_y']
                                ).rename(columns={'count_x':'debits_count', 
                                                'sum_x':'debits_sum', 
                                                'count_y':'credits_count', 
                                                'sum_y':'credits_sum'}).fillna(0)
    
    deb_cred_net['net_count'] = deb_cred_net['debits_count'] + deb_cred_net['credits_count']
    deb_cred_net['net_sum'] = deb_cred_net['debits_sum'] + deb_cred_net['credits_sum']
    
    return deb_cred_net

def main():
    ################################################################################
    # Get all data assets to parquet files

    # Used for all versions of the data asset
    COICOP_labels_df, COICOP_dic = get_COICOP_maps()
    top_trim_quantile = 0.9975

    ################################################################################
    # uk data first
    market = 'uk'

    is_net = False
    print(f'Getting gross data for {market}')
    adf_gross, adf_deb_and_cred = get_asset(db_pth_uk, COICOP_labels_df, is_net = is_net, market=market)
    print('Pivoting')
    pdf_gross = get_pivoted_data(adf_gross, 'COICOP_14.0', 'gambling', top_trim_quantile=top_trim_quantile)
    quantiles=pdf_gross[['user_id','month', 'decile_abs', 'quintile_abs']].drop_duplicates()

    is_net = True
    print(f'Getting net data for {market}')
    adf_net, adf_deb_and_cred = get_asset(db_pth_uk, COICOP_labels_df, is_net = is_net, market=market)
    print('Pivoting')
    pdf_net = get_pivoted_data(adf_net, 'COICOP_14.0', 'gambling', quantiles=quantiles)

    # For descriptives
    deb_cred_net = get_debits_credits_and_net_by_decile(adf_deb_and_cred, pdf_net)

    # Population level
    print(f'Getting full population data for {market}')
    adf_pop_uk, adf_deb_and_cred_pop_gb = get_asset(db_pth_uk, COICOP_labels_df, is_net = is_net, market=market, drop_non_gamblers=False)
    print('Pivoting')
    pdf_pop_uk = get_pivoted_data(adf_pop_uk, 'COICOP_14.0', 'gambling')

    # Version with coffee extracted
    print('Extracting coffe from UK net data')
    adf_coffee = pull_coffee_into_new_category(db_pth_uk, adf_net, market)
    pdf_coffee = get_pivoted_data(adf_coffee, 'COICOP_16.0', 'on_coffee', top_trim_quantile=top_trim_quantile)

    # Output data asset(s)
    print('Outputting all UK data assets')
    adf_net.to_parquet(asset_pq_uk)
    pdf_net.to_parquet(pdf_pth_uk)

    adf_gross.to_parquet(asset_pq_gross_uk)
    pdf_gross.to_parquet(pdf_pth_gross_uk)

    adf_deb_and_cred.to_parquet(asset_deb_and_cred_pq_uk)
    deb_cred_net.to_parquet(asset_deb_cred_net_pq_uk)
        
    adf_pop_uk.to_parquet(pop_asset_pq_uk)
    pdf_pop_uk.to_parquet(pop_pdf_pth_uk)

    adf_coffee.to_parquet(asset_pq_uk_coffee)
    pdf_coffee.to_parquet(pdf_pth_uk_coffee)

    ################################################################################
    # Now US data
    market = 'US'

    is_net = False
    print(f'Getting gross data for {market}')
    adf_gross, adf_deb_and_cred = get_asset(db_pth_us, COICOP_labels_df, is_net = is_net, market=market)
    print('Pivoting')
    pdf_gross = get_pivoted_data(adf_gross, 'COICOP_14.0', 'gambling', top_trim_quantile=top_trim_quantile)
    quantiles=pdf_gross[['user_id','month', 'decile_abs', 'quintile_abs']].drop_duplicates()

    is_net = True
    print(f'Getting net data for {market}')
    adf_net, adf_deb_and_cred = get_asset(db_pth_us, COICOP_labels_df, is_net = is_net, market=market)
    print('Pivoting')
    pdf_net = get_pivoted_data(adf_net, 'COICOP_14.0', 'gambling', quantiles=quantiles)

    # For descriptives
    deb_cred_net = get_debits_credits_and_net_by_decile(adf_deb_and_cred, pdf_net)

    # Population 
    print(f'Getting full population data for {market}')
    adf_pop_us, adf_deb_and_cred_pop_us = get_asset(db_pth_us, COICOP_labels_df, is_net = is_net, market=market, drop_non_gamblers=False)
    print('Pivoting')
    pdf_pop_us = get_pivoted_data(adf_pop_us, 'COICOP_14.0', 'gambling')

    # Output data asset(s)
    print('Outputting all US data assets')
    adf_net.to_parquet(asset_pq_us)
    pdf_net.to_parquet(pdf_pth_us)

    adf_gross.to_parquet(asset_pq_gross_us)
    pdf_gross.to_parquet(pdf_pth_gross_us)

    adf_deb_and_cred.to_parquet(asset_deb_and_cred_pq_us)
    deb_cred_net.to_parquet(asset_deb_cred_net_pq_us)
    
    adf_pop_us.to_parquet(pop_asset_pq_us)
    pdf_pop_us.to_parquet(pop_pdf_pth_us)

if __name__ == "__main__":
    main()