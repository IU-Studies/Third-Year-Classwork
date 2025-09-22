

## **1. Count Documents in a Collection**

```js
// Example documents
db.myCollection.insertMany([
    { name: "Doc1" },
    { name: "Doc2" },
    { name: "Doc3" }
]);

// MapReduce
db.myCollection.mapReduce(
    function() { emit("count", 1); },
    function(key, values) { return Array.sum(values); },
    { out: "document_count" }
);

// View result
db.document_count.find();
```

✅ Output:

```json
{ "_id": "count", "value": 3 }
```

---

## **2. Word Count**

```js
// Example documents
db.texts.insertMany([
    { _id: 1, content: "hello world hello" },
    { _id: 2, content: "world of mongodb" }
]);

// MapReduce
db.texts.mapReduce(
    function() {
        var words = this.content.split(" ");
        for (var i = 0; i < words.length; i++) {
            emit(words[i], 1);
        }
    },
    function(key, values) {
        return Array.sum(values);
    },
    { out: "word_count" }
);

// View result
db.word_count.find();
```

✅ Output:

```json
{ "_id": "hello", "value": 2 }
{ "_id": "world", "value": 2 }
{ "_id": "of", "value": 1 }
{ "_id": "mongodb", "value": 1 }
```

---

## **3. Sum of Values by Category**

```js
// Example documents
db.sales.insertMany([
    { _id: 1, category: "Electronics", amount: 500 },
    { _id: 2, category: "Books", amount: 200 },
    { _id: 3, category: "Electronics", amount: 300 }
]);

// MapReduce
db.sales.mapReduce(
    function() { emit(this.category, this.amount); },
    function(key, values) { return Array.sum(values); },
    { out: "category_sales" }
);

// View result
db.category_sales.find();
```

✅ Output:

```json
{ "_id": "Electronics", "value": 800 }
{ "_id": "Books", "value": 200 }
```

---

## **4. Average Marks per Subject**

```js
// Example documents
db.marks.insertMany([
    { _id: 1, subject: "Math", marks: 80 },
    { _id: 2, subject: "Math", marks: 90 },
    { _id: 3, subject: "Science", marks: 70 }
]);

// MapReduce
db.marks.mapReduce(
    function() { emit(this.subject, { sum: this.marks, count: 1 }); },
    function(key, values) {
        var result = { sum: 0, count: 0 };
        values.forEach(function(v) {
            result.sum += v.sum;
            result.count += v.count;
        });
        return result;
    },
    {
        out: "subject_avg",
        finalize: function(key, reducedValue) {
            return reducedValue.sum / reducedValue.count;
        }
    }
);

// View result
db.subject_avg.find();
```

✅ Output:

```json
{ "_id": "Math", "value": 85 }
{ "_id": "Science", "value": 70 }
```
