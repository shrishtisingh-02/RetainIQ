-- Queries run against an in-memory SQLite table named `transactions`
-- (the cleaned data: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
--  UnitPrice, CustomerID, Country, Revenue).
-- Each block starts with "-- name: <query_name>".

-- name: monthly_revenue
SELECT strftime('%Y-%m', InvoiceDate)  AS month,
       ROUND(SUM(Revenue), 2)          AS revenue,
       COUNT(DISTINCT InvoiceNo)       AS orders,
       COUNT(DISTINCT CustomerID)      AS customers
FROM transactions
GROUP BY month
ORDER BY month;

-- name: top_countries
SELECT Country,
       ROUND(SUM(Revenue), 2)     AS revenue,
       COUNT(DISTINCT CustomerID) AS customers,
       COUNT(DISTINCT InvoiceNo)  AS orders
FROM transactions
GROUP BY Country
ORDER BY revenue DESC
LIMIT 10;

-- name: top_products
SELECT StockCode,
       MAX(Description)       AS description,
       SUM(Quantity)          AS units_sold,
       ROUND(SUM(Revenue), 2) AS revenue
FROM transactions
GROUP BY StockCode
ORDER BY revenue DESC
LIMIT 15;

-- name: repeat_purchase_rate
WITH orders_per_customer AS (
    SELECT CustomerID, COUNT(DISTINCT InvoiceNo) AS n_orders
    FROM transactions
    GROUP BY CustomerID
)
SELECT COUNT(*)                                             AS customers,
       SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)        AS repeat_customers,
       ROUND(100.0 * SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                            AS repeat_rate_pct
FROM orders_per_customer;

-- name: revenue_concentration
WITH customer_revenue AS (
    SELECT CustomerID, SUM(Revenue) AS revenue
    FROM transactions
    GROUP BY CustomerID
),
ranked AS (
    SELECT revenue, NTILE(5) OVER (ORDER BY revenue DESC) AS quintile
    FROM customer_revenue
)
SELECT quintile,
       COUNT(*)                                                    AS customers,
       ROUND(SUM(revenue), 2)                                      AS revenue,
       ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM ranked), 2) AS revenue_share_pct
FROM ranked
GROUP BY quintile
ORDER BY quintile;
