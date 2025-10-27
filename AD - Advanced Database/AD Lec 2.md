## Exp 2 - Consider any organizational database schema and implement all MongoDB methods on it.


# 🏥 Hospital Database using MongoDB

This project demonstrates how to use **MongoDB** for managing a hospital database.
It includes examples of **CRUD operations**, **queries**, **sorting**, **aggregation**, and **indexing**.

---

## 📂 Database & Collection Setup

```js
use hospital

db.patients.insertMany([
  {
    patient_id: "P001",
    name: "IU",
    age: 25,
    phno: "9876543210",
    disease: "Flu",
    allocated_doctor: "Dr. Smith",
    medicine: ["Paracetamol", "Cough Syrup"],
    visit_date: new Date("2025-08-01")
  },
  {
    patient_id: "P002",
    name: "UI",
    age: 30,
    phno: "9876501234",
    disease: "Diabetes",
    allocated_doctor: "Dr. Emily",
    medicine: ["Insulin", "Metformin"],
    visit_date: new Date("2025-08-03")
  },
  {
    patient_id: "P003",
    name: "MEIU",
    age: 40,
    phno: "9845612345",
    disease: "Hypertension",
    allocated_doctor: "Dr. Ray",
    medicine: ["Amlodipine"],
    visit_date: new Date("2025-07-29")
  },
  {
    patient_id: "P004",
    name: "MIKU",
    age: 22,
    phno: "9823456789",
    disease: "Asthma",
    allocated_doctor: "Dr. Chloe",
    medicine: ["Inhaler"],
    visit_date: new Date("2025-08-05")
  },
  {
    patient_id: "P005",
    name: "MAU",
    age: 28,
    phno: "9811122233",
    disease: "Migraine",
    allocated_doctor: "Dr. Harris",
    medicine: ["Ibuprofen"],
    visit_date: new Date("2025-08-10")
  },
  {
    patient_id: "P006",
    name: "Cipher",
    age: 35,
    phno: "9898989898",
    disease: "Covid-19",
    allocated_doctor: "Dr. Alice",
    medicine: ["Remdesivir", "Vitamin C"],
    visit_date: new Date("2025-08-12")
  },
  {
    patient_id: "P007",
    name: "HarPar",
    age: 45,
    phno: "9777777777",
    disease: "Arthritis",
    allocated_doctor: "Dr. Noah",
    medicine: ["Naproxen"],
    visit_date: new Date("2025-08-15")
  },
  {
    patient_id: "P008",
    name: "OR2",
    age: 50,
    phno: "9666666666",
    disease: "Heart Disease",
    allocated_doctor: "Dr. Zoe",
    medicine: ["Aspirin", "Atorvastatin"],
    visit_date: new Date("2025-08-16")
  }
]);
```

---

## 🔍 Queries

### 2.1 Find all documents

```js
db.patients.find();
```

👉 Retrieves **all patients** from the collection.

### 2.2 Find one document

```js
db.patients.findOne({ patient_id: "P001" });
```

👉 Returns the **first matching document** where `patient_id = "P001"`.

### 2.3 Find with condition

```js
db.patients.find({ disease: "Diabetes" });
```

👉 Fetches all patients diagnosed with **Diabetes**.

### 2.4 Find with multiple conditions

```js
db.patients.find({ age: { $gt: 30 }, disease: "Hypertension" });
```

👉 Finds patients **older than 30** who also have **Hypertension**.

### 2.5 Projection (select fields)

```js
db.patients.find({}, { name: 1, disease: 1, _id: 0 });
```

👉 Displays only **name** and **disease**, hides `_id`.

### 2.6 Count documents

```js
db.patients.countDocuments();
```

👉 Returns the **total number of patients**.

---

## ✏️ Update Operations

### 3.1 Update one document

```js
db.patients.updateOne(
  { patient_id: "P001" },
  { $set: { phno: "9000000000" } }
);
```

👉 Updates the **phone number** of patient `P001`.

### 3.2 Update many documents

```js
db.patients.updateMany({}, { $inc: { age: 1 } });
```

👉 Increments the **age of all patients** by `1`.

---

## 🗑️ Delete Operations

### 4.1 Delete one document

```js
db.patients.deleteOne({ patient_id: "P010" });
```

👉 Deletes a patient with `patient_id = "P010"` (if exists).

### 4.2 Delete many documents

```js
db.patients.deleteMany({ age: { $lt: 25 } });
```

👉 Removes all patients **younger than 25 years**.

### 4.3 Drop entire collection

```js
db.patients.drop();
```

👉 **Deletes the whole patients collection** permanently.

---

## 📊 Sorting & Pagination

### 5.1 Sort by age (ascending)

```js
db.patients.find().sort({ age: 1 });
```

### 5.2 Sort by visit\_date (descending)

```js
db.patients.find().sort({ visit_date: -1 });
```

### 5.3 Limit results

```js
db.patients.find().limit(3);
```

👉 Shows only **first 3 patients**.

### 5.4 Skip + limit (pagination)

```js
db.patients.find().skip(2).limit(3);
```

👉 Skips **first 2** patients and returns the **next 3**.

---

## ⚡ Indexing

### 6.1 Create index

```js
db.patients.createIndex({ patient_id: 1 });
```

👉 Creates an **index on patient\_id** to speed up searches.

### 6.2 View indexes

```js
db.patients.getIndexes();
```

👉 Lists all indexes created on the collection.

---

## 📈 Aggregation Framework

### 7.1 Group by disease and count patients

```js
db.patients.aggregate([
  { $group: { _id: "$disease", total: { $sum: 1 } } }
]);
```

👉 Counts **number of patients per disease**.

### 7.2 Average age per doctor

```js
db.patients.aggregate([
  { $group: { _id: "$allocated_doctor", avg_age: { $avg: "$age" } } }
]);
```

👉 Shows **average patient age** for each doctor.

### 7.3 Match + Project (patients above 30)

```js
db.patients.aggregate([
  { $match: { age: { $gt: 30 } } },
  { $project: { name: 1, age: 1, _id: 0 } }
]);
```

👉 Finds **patients older than 30** and shows only `name` & `age`.

