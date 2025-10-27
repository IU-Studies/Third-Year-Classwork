## Exp 4 - Distributed Database: Implementation of Partitions: Round robin, Range partitioning techniques, List using Relational Databases.

# Range Partitioning

### **Table Creation Queries**

```sql
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
```


### **Partition Creation Queries**

```sql
create table student_range_low partition of student_range for values from (0) to (40);
create table student_range_mid partition of student_range for values from (40) to (70);
create table student_range_high partition of student_range for values from (70) to (101);
```


### **Data Insertion Queries**

```sql
insert into student_range values (1, 'IU', 'CSE', 55);
insert into student_range values (2, 'UI', 'IT', 99);
insert into student_range values (3, 'IUI', 'MECH', 32);
insert into student_range values (4, 'IUU', 'ENTC', 40);
```


### **Select Queries**

```sql
select * from student_range_low;
select * from student_range_mid;
select * from student_range_high;
```

# List Partitioning


### **Table Creation Queries**

```sql
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
```


### **Partition Creation Queries**

```sql
create table student_cs partition of student_list for values in ('CSE');
create table student_it partition of student_list for values in ('IT');
create table student_etc partition of student_list for values in ('ENTC');
create table student_other partition of student_list default;
```


### **Data Insertion Queries**

```sql
insert into student_list values (1, 'IU', 'CSE', 23);
insert into student_list values (2, 'ICU', 'CSE', 23);
insert into student_list values (3, 'EIU', 'IT', 23);
insert into student_list values (4, 'IBU', 'ENTC', 23);
insert into student_list values (5, 'IWU', 'MECH', 23);
```

### **Select Queries**

```sql
select * from student_cs;
select * from student_it;
select * from student_etc;
select * from student_other;
```

# Round Robin



### **Table & Partition Setup**

```sql
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
```


### **Sequence & Trigger Function**

```sql
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
```


### **Trigger Creation**

```sql
CREATE TRIGGER rr_trigger
BEFORE INSERT ON student_rr
FOR EACH ROW EXECUTE FUNCTION rr_insert();
```


### **Insert Queries**

```sql
INSERT INTO student_rr VALUES (8, 'Varun', 'CSE', 50);
INSERT INTO student_rr VALUES (9, 'Neha', 'IT', 75);
INSERT INTO student_rr VALUES (10, 'Suresh', 'ENTC', 30);
```


### **Select Queries**

```sql
SELECT * FROM student_rr_p1;
SELECT * FROM student_rr_p2;
```
