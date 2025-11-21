-- Get non-savings accounts. Deal with savings in separate query
    WITH accs AS (SELECT id AS account_id
                        FROM accounts 
                        WHERE NOT (sav_check)
    ),
    
    -- Get transactions that are not internal to the user
    MonthTransactions AS (
        SELECT 
            t.user_id,
            strftime('%Y-%m', t.timestamp) AS month, 
            t.amount,
            t.COICOP,
        FROM transactions t JOIN accs a
        ON t.account_id=a.account_id
        WHERE (NOT t.is_transaction_with_self)
    ),

    -- Calculate min and max months per user to CTE
    MinMax AS (
        SELECT 
            user_id, 
            MIN(month) AS uid_min_month, 
            MAX(month) AS uid_max_month
        FROM MonthTransactions
        GROUP BY user_id
    ),

    -- Get rid of first and last month per user
    Filtered AS (SELECT t.*,
                        m.uid_min_month,
                        m.uid_max_month,
                        t.amount < 0 AS is_debit
                FROM MonthTransactions t
                JOIN MinMax m ON t.user_id = m.user_id
                WHERE t.month NOT IN (m.uid_min_month, m.uid_max_month)
    ),

    -- Aggregate data
    FinalOutput AS (
        SELECT 
            user_id,
            month,
            is_debit,
            COALESCE(COICOP, 0) AS COICOP,  -- Just in case there are nulls
            COUNT(amount) AS count,
            SUM(amount) AS sum
        FROM Filtered
        GROUP BY 
            user_id, month, COICOP, is_debit
    )

    -- Output transactions that are not unclassified, are not classed as a transfer to credit card or savings 
    --   (dealt with in separate queries)
    SELECT * FROM FinalOutput WHERE NOT (COICOP IN (12.21, 12.22));