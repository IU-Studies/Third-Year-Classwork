import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["YIBD"]
collection = db["Students"]

# 1. Insertion of Data
data = { "name": "IU", "address": "New York" }
collection.insert_one(data)

many_data = [
    { "name": "Meku", "address": "California" },
    { "name": "Mini", "address": "Texas" },
    { "name": "HarPar", "address": "Nevada" }
]

collection.insert_many(many_data)


# 2. Reading of Data
print("Print 1st Value")
print(collection.find_one())

print(" ")
print("Printing all document one by one: ")
for i in collection.find():
    print(i)


# 3. Updating Document
print(" ")
print("Printing updated document: ")
collection.update_one(
    { "name": "IU" },
    { "$set": { "address": "Chicago" } }
)
print(collection.find_one({ "name": "IU" }))


# 4. Deleting Document
print(" ")
print("Printing all document one by one after deletion: ")
collection.delete_one({ "name": "Meku" })
for i in collection.find():
    print(i)
