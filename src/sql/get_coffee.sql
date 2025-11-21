WITH accs AS ( -- Only transactions from non-savings accounts
            SELECT id AS account_id
            FROM accounts 
            WHERE NOT (sav_check)
            )                                      
                    
SELECT t.user_id,
    strftime('%Y-%m', t.timestamp) AS month, 
    COALESCE(t.COICOP, 0) AS COICOP,  -- Just in case there are nulls
    COUNT(t.amount) AS count,
    SUM(t.amount) AS sum
FROM transactions t JOIN accs a
ON t.account_id=a.account_id
WHERE t.category='Coffee Shops' 
AND t.amount<0 
GROUP BY t.user_id, month, t.COICOP