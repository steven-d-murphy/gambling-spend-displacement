    -- Only credit card accounts
    WITH cc_accounts AS (SELECT id AS account_id,
                        FROM accounts 
                        WHERE credit_check),

    -- Get CC interest payments
    cc_int_payments AS (SELECT t.user_id, strftime('%Y-%m', t.timestamp) AS month, t.amount
                        FROM transactions t INNER JOIN cc_accounts a 
                        ON t.account_id=a.account_id
                        WHERE t.amount<0.0 AND t.brand LIKE '%interest%'),
    
    -- Aggregate
    FinalOutput AS (SELECT user_id, month, 12.21 AS COICOP, COUNT(amount) AS count, SUM(amount) AS sum
        FROM cc_int_payments
        GROUP BY user_id, month)

    SELECT * FROM FinalOutput;