

## Code 1: Table Creation and Data Insertion

```sql
CREATE TABLE Sales (
  Sale_ID NUMBER PRIMARY KEY,
  Region VARCHAR2(20),
  Product VARCHAR2(20),
  Sale_Date DATE,
  Quantity NUMBER,
  Amount NUMBER
);

INSERT INTO Sales VALUES (1, 'North', 'Laptop', DATE '2025-01-10', 5, 250000);
INSERT INTO Sales VALUES (2, 'North', 'Mobile', DATE '2025-01-15', 10, 150000);
INSERT INTO Sales VALUES (3, 'South', 'Laptop', DATE '2025-02-05', 8, 400000);
INSERT INTO Sales VALUES (4, 'South', 'Tablet', DATE '2025-02-10', 6, 180000);
INSERT INTO Sales VALUES (5, 'East', 'Laptop', DATE '2025-03-01', 4, 220000);
INSERT INTO Sales VALUES (6, 'East', 'Mobile', DATE '2025-03-02', 7, 220000);
INSERT INTO Sales VALUES (7, 'West', 'Tablet', DATE '2025-03-15', 9, 300000);
INSERT INTO Sales VALUES (8, 'West', 'Mobile', DATE '2025-03-18', 11, 150000);
COMMIT;
```

**Explanation:**
Creates a `Sales` table to store transaction details such as region, product, and sales amount.
Inserts sample sales data for multiple regions and products with corresponding amounts.

---

## Code 2: ROLLUP Query

```sql
SELECT Region, Product, SUM(Amount) AS Total_Sales
FROM Sales
GROUP BY ROLLUP (Region, Product);
```

**Explanation:**
Generates subtotal and grand total sales by region and product using `ROLLUP`.
Displays totals per product, per region, and an overall total.

---

## Code 3: CUBE Query

```sql
SELECT Region, Product, SUM(Amount) AS Total_Sales
FROM Sales
GROUP BY CUBE (Region, Product);
```

**Explanation:**
Calculates all possible subtotal combinations using `CUBE`.
Provides totals per product, per region, and their crosswise combinations for analysis.

---

## Code 4: RANK

```sql
SELECT Region, Product, Amount,
RANK() OVER (PARTITION BY Region ORDER BY Amount DESC) AS RankInRegion
FROM Sales;
```

**Explanation:**
Assigns rankings to products within each region based on sales amount.
Skips rank numbers when ties occur (e.g., 1, 2, 2, 4).

---

## Code 5: DENSE_RANK

```sql
SELECT Region, Product, Amount,
DENSE_RANK() OVER (PARTITION BY Region ORDER BY Amount DESC) AS Ranks
FROM Sales;
```

**Explanation:**
Similar to `RANK()` but does not skip rank numbers after ties.
Consecutive ranks are assigned (e.g., 1, 2, 2, 3).

---

## Code 6: FIRST_VALUE / LAST_VALUE

```sql
SELECT Region,
FIRST_VALUE(Product) OVER (PARTITION BY Region ORDER BY Sale_Date) AS First_Product,
LAST_VALUE(Product) OVER (PARTITION BY Region ORDER BY Sale_Date
  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS Last_Product
FROM Sales;
```

**Explanation:**
Retrieves the first and last sold product per region based on `Sale_Date`.
Uses window framing to include all rows within each region.

---
