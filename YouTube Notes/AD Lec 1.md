
# 📘 MongoDB Basic Commands 


## 🗂️ 1. Create / Switch Database

```js
use myDatabase
```

**English:** Switches to `myDatabase`. If it doesn’t exist, MongoDB will create it automatically.

**Hinglish:** `use` command se hum kisi bhi database me switch karte hain — agar wo database already hai, toh switch ho jaayega, nahi hai toh naya create ho jaayega.

---

## 📁 2. Create a Collection

```js
db.createCollection("users")
```

**English:** Creates a new collection named `users` inside the current database.

**Hinglish:** Ye command current database me `users` naam ka ek naya collection banati hai. Collection matlab table jaisa structure.

---

## 📝 3. Insert Data

### ➤ Insert One Document

```js
db.users.insertOne({ name: "IU", age: 25 })
```

**English:** Inserts a single document (record) into the `users` collection.

**Hinglish:** Ye command ek hi record insert karti hai `users` collection ke andar.

### ➤ Insert Many Documents

```js
db.users.insertMany([
  { name: "Cipher", age: 24 },
  { name: "HarPar", age: 28 }
])
```

**English:** Inserts multiple documents at once into the collection.

**Hinglish:** Ek se zyada documents ek hi baar me insert karne ke liye `insertMany` use karte hain.

---

## 🔍 4. Find / Retrieve Data

### ➤ Find All Documents

```js
db.users.find()
```

**English:** Returns all documents present in the `users` collection.

**Hinglish:** Collection ke sabhi records ko dekhne ke liye `find()` ka use hota hai.

### ➤ Find with Condition

```js
db.users.find({ age: 25 })
```

**English:** Finds documents where age is 25.

**Hinglish:** Yahaan hum condition laga rahe hain — sirf unhi documents ko fetch karo jinki age 25 hai.

---

## ✏️ 5. Update Data

```js
db.users.updateOne(
  { name: "IU" },
  { $set: { age: 26 } }
)
```

**English:** Updates the age of the first document where the name is "IU".

**Hinglish:** `updateOne` sirf pehle matching record ko update karta hai — yahaan hum IU ki age 26 kar rahe hain.

---

## ❌ 6. Delete Data

### ➤ Delete One Document

```js
db.users.deleteOne({ name: "HarPar" })
```

**English:** Deletes the first document that matches the condition.

**Hinglish:** `deleteOne` sirf pehla matching document delete karta hai — yahan `HarPar` naam wala record delete ho jaayega.

### ➤ Delete Multiple Documents

```js
db.users.deleteMany({ age: { $gt: 30 } })
```

**English:** Deletes all documents where age is greater than 30.

**Hinglish:** Jo log 30 se zyada age ke hain, un sabhi records ko delete karne ke liye `deleteMany` ka use karte hain.

---

## 📋 7. Show Collections and Databases

### ➤ Show All Databases

```js
show dbs
```

**English:** Lists all databases available in MongoDB.

**Hinglish:** MongoDB ke andar jitne bhi databases hain, sabki list dikhata hai.

### ➤ Show All Collections in Current DB

```js
show collections
```

**English:** Shows all collections inside the current database.

**Hinglish:** Current database ke andar jitne bhi collections hain, unki list show karta hai.

