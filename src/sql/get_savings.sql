   
    -- Get net savings
    WITH sav_accounts AS (SELECT id AS account_id,
                        FROM accounts 
                        WHERE sav_check),
    
    non_sav_accounts AS (SELECT id AS account_id,
                        FROM accounts 
                        WHERE NOT (sav_check)),

    -- We want all transactions to and from savings accounts. Get the -ve of the amount
    -- because +ve payments into the account are 'spend' on savings and our convention
    -- up to now is to have spend as a -ve
    sav_account_payments AS (SELECT t.user_id, strftime('%Y-%m', t.timestamp) AS month, -t.amount AS amount
                        FROM transactions t INNER JOIN sav_accounts a 
                        ON t.account_id=a.account_id),

    -- Get non-internal 'spend' from non-savings accounts marked as saving (i.e., savings accounts that we can't see)
    non_sav_account_payments AS (SELECT t.user_id, strftime('%Y-%m', t.timestamp) AS month, t.amount
                        FROM transactions t INNER JOIN non_sav_accounts a 
                        ON t.account_id=a.account_id
                        WHERE t.COICOP==12.22),

    -- Append the two tables of transactions to find whether there is saving
    all_payments AS (SELECT user_id, month, amount
                        FROM sav_account_payments
                        UNION ALL
                        SELECT user_id, month, amount
                        FROM non_sav_account_payments),

    OutputTable AS (SELECT user_id, month, 12.22 AS COICOP, COUNT(amount) AS sav_count, SUM(amount) AS sav_sum
                    FROM all_payments
                    GROUP BY user_id, month)
   
    -- Output a row for all user_id+months saving takes place
    SELECT * FROM OutputTable WHERE sav_sum < 0.0; 