```sql
-- Exp 1

create table student (
    student_id INT primary key,
    student_name varchar(50),
    department varchar(20),
    marks INT
);

create table student_range (
    student_id INT,
    student_name varchar(50),
    department varchar(20),
    marks INT
) partition by range(marks);

create table student_range_low partition of student_range for values from (0) to (40);
create table student_range_mid partition of student_range for values from (40) to (70);
create table student_range_high partition of student_range for values from (70) to (101);

insert into student_range values (1, 'IU', 'CSE', 55);
insert into student_range values (2, 'UI', 'IT', 99);
insert into student_range values (3, 'IUI', 'MECH', 32);
insert into student_range values (4, 'IUU', 'ENTC', 40);

select * from student_range_low;
select * from student_range_mid;
select * from student_range_high;



create table student (
    student_id INT primary key,
    student_name varchar(50),
    department varchar(20),
    marks INT
);

create table student_list (
    student_id INT,
    student_name varchar(50),
    department varchar(20),
    marks INT
) partition by list (department);

create table student_cs partition of student_list for values in ('CSE');
create table student_it partition of student_list for values in ('IT');
create table student_etc partition of student_list for values in ('ENTC');
create table student_other partition of student_list default;

insert into student_list values (1, 'IU', 'CSE', 23);
insert into student_list values (2, 'ICU', 'CSE', 23);
insert into student_list values (3, 'EIU', 'IT', 23);
insert into student_list values (4, 'IBU', 'ENTC', 23);
insert into student_list values (5, 'IWU', 'MECH', 23);

select * from student_cs;
select * from student_it;
select * from student_etc;
select * from student_other;



CREATE TABLE student (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(50),
    department VARCHAR(20),
    marks INT
);

CREATE TABLE student_rr (
    student_id INT,
    student_name VARCHAR(50),
    department VARCHAR(20),
    marks INT
);

CREATE TABLE student_rr_p1 (LIKE student_rr);
CREATE TABLE student_rr_p2 (LIKE student_rr);

CREATE SEQUENCE rr_seq START 1;

CREATE OR REPLACE FUNCTION rr_insert()
RETURNS TRIGGER AS $$
BEGIN
    IF (nextval('rr_seq') % 2 = 1) THEN
        INSERT INTO student_rr_p1 VALUES (NEW.*);
    ELSE
        INSERT INTO student_rr_p2 VALUES (NEW.*);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER rr_trigger
BEFORE INSERT ON student_rr
FOR EACH ROW EXECUTE FUNCTION rr_insert();

INSERT INTO student_rr VALUES (8, 'Varun', 'CSE', 50);
INSERT INTO student_rr VALUES (9, 'Neha', 'IT', 75);
INSERT INTO student_rr VALUES (10, 'Suresh', 'ENTC', 30);

SELECT * FROM student_rr_p1;
SELECT * FROM student_rr_p2;






-- Exp 2
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

SELECT Region, Product, SUM(Amount) AS Total_Sales
FROM Sales
GROUP BY ROLLUP (Region, Product);

SELECT Region, Product, SUM(Amount) AS Total_Sales
FROM Sales
GROUP BY CUBE (Region, Product);

SELECT Region, Product, Amount,
RANK() OVER (PARTITION BY Region ORDER BY Amount DESC) AS RankInRegion
FROM Sales;

SELECT Region, Product, Amount,
DENSE_RANK() OVER (PARTITION BY Region ORDER BY Amount DESC) AS DenseRanks
FROM Sales;

SELECT Region,
FIRST_VALUE(Product) OVER (PARTITION BY Region ORDER BY Sale_Date) AS First_Product,
LAST_VALUE(Product) OVER (PARTITION BY Region ORDER BY Sale_Date
  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS Last_Product
FROM Sales;



-- Exp 3 - Design and implement a data warehouse for an online retail company to analyze sales across products, regions, and time. Create the warehouse using Star Schema, Snowflake Schema, and Fact Constellation Schema to compare different designs. 
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
```


Step 1: Import Dataset
1. Open RapidMiner Studio.
2. Go to the &#39;Repository&#39; panel and right-click → Import Data.
3. Select the sales transaction dataset (e.g., CSV/Excel file).
4. Verify the attributes and metadata (e.g., date, product, region, sales_amount).
Step 2: Data Preprocessing
1. Handle Missing Values:
- Use the &#39;Replace Missing Values&#39; operator.
- Choose appropriate strategies (mean, median, mode, or remove rows).
2. Normalization:
- Use the &#39;Normalize&#39; operator to scale numeric features.
- This ensures all variables contribute equally to the model.
3. Attribute Selection:
- Use &#39;Select Attributes&#39; to choose relevant features (e.g., product category, region, month, past
sales).
Step 3: Build Predictive Model
1. Choose a Machine Learning Algorithm:
- Linear Regression (for continuous sales prediction).
- Decision Tree / Random Forest (for pattern-based prediction).
- Neural Networks (for complex data with nonlinear relationships).
2. Add the chosen operator (e.g., &#39;Linear Regression&#39;) to the process.

3. Connect the preprocessed dataset to the model operator.
Step 4: Model Evaluation
1. Use the &#39;Split Data&#39; operator to divide data into Training (70%) and Testing (30%).
2. Apply &#39;Performance&#39; operator to evaluate the model.
3. Metrics to check:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Correlation (R²)
4. Compare results across different models (e.g., Linear Regression vs Random Forest).
Step 5: Sales Forecasting
1. Once the best model is chosen, apply it on new unseen data (future dates).
2. Use the model predictions to estimate future sales trends.
3. Export results for reporting and decision-making.
