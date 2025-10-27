-- Step 1: Create Dimension Tables
CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    full_date DATE,
    month INT,
    quarter INT,
    year INT
);

CREATE TABLE dim_product (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(50),
    category VARCHAR(50),
    subcategory VARCHAR(50)
);

CREATE TABLE dim_region (
    region_id INT PRIMARY KEY,
    country VARCHAR(50),
    state VARCHAR(50),
    city VARCHAR(50)
);

-- Step 2: Create Fact Table (Star Schema)
CREATE TABLE fact_sales (
    sale_id INT PRIMARY KEY,
    date_id INT,
    product_id INT,
    region_id INT,
    units_sold INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (region_id) REFERENCES dim_region(region_id)
);

-- Step 3: Insert Sample Data
-- Date Dimension
INSERT INTO dim_date VALUES (1, '2024-01-01', 1, 1, 2024);
INSERT INTO dim_date VALUES (2, '2024-02-01', 2, 1, 2024);
INSERT INTO dim_date VALUES (3, '2025-01-01', 1, 1, 2025);

-- Product Dimension
INSERT INTO dim_product VALUES (1, 'Laptop', 'Electronics', 'Computers');
INSERT INTO dim_product VALUES (2, 'Mouse', 'Electronics', 'Accessories');
INSERT INTO dim_product VALUES (3, 'Headphone', 'Electronics', 'Audio');

-- Region Dimension
INSERT INTO dim_region VALUES (1, 'India', 'Maharashtra', 'Pune');
INSERT INTO dim_region VALUES (2, 'India', 'Karnataka', 'Bengaluru');

-- Fact Table
INSERT INTO fact_sales VALUES (1, 1, 1, 1, 2, 50000, 100000);
INSERT INTO fact_sales VALUES (2, 1, 2, 1, 5, 500, 2500);
INSERT INTO fact_sales VALUES (3, 2, 3, 2, 10, 1000, 10000);
INSERT INTO fact_sales VALUES (4, 3, 1, 2, 1, 52000, 52000);

-- Step 4: Create Fact Constellation Table (Returns)
CREATE TABLE fact_returns (
    return_id INT PRIMARY KEY,
    sale_id INT,
    date_id INT,
    product_id INT,
    region_id INT,
    units_returned INT,
    return_amount DECIMAL(10,2),
    FOREIGN KEY (sale_id) REFERENCES fact_sales(sale_id)
);

INSERT INTO fact_returns VALUES (1, 2, 2, 2, 1, 1, 500);

-- OLAP Queries

-- 1) Total Sales by Year and Region — Using ROLLUP
SELECT d.year, r.state, SUM(f.total_amount) AS total_sales
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_region r ON f.region_id = r.region_id
GROUP BY ROLLUP (d.year, r.state);

-- 2) Total Sales by Product and Year — Using CUBE
SELECT p.product_name, d.year, SUM(f.total_amount) AS total_sales
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY CUBE (p.product_name, d.year);

-- 3) Rank Top Products by Total Sales
SELECT p.product_name,
       SUM(f.total_amount) AS total_sales,
       RANK() OVER (ORDER BY SUM(f.total_amount) DESC) AS sales_rank
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY p.product_name;

-- 4) Sales vs Returns (Fact Constellation Analysis)
SELECT p.product_name,
       SUM(s.total_amount) AS total_sales,
       COALESCE(SUM(r.return_amount), 0) AS total_returns,
       (SUM(s.total_amount) - COALESCE(SUM(r.return_amount), 0)) AS net_sales
FROM fact_sales s
LEFT JOIN fact_returns r ON s.sale_id = r.sale_id
JOIN dim_product p ON s.product_id = p.product_id
GROUP BY p.product_name;

-- 5) Sales Trend by Year and Region
SELECT d.year, r.state, SUM(f.total_amount) AS total_sales
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_region r ON f.region_id = r.region_id
GROUP BY d.year, r.state
ORDER BY d.year;
