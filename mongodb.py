from pymongo import MongoClient

cluster ="mongodb+srv://buwanij:bV2j7q42hxuyRj2l@cluster0.vc0zpcu.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(cluster)

# print(client.list_database_names())

db = client.shops

# print(db.list_collection_names())

itms = db.itms

# item2 = [{"name":"dress","price":"rs.10.00","colors":["red","black"]},
#          {"name":"short","price":"rs.20.00","colors":["red","black"]},
#          {"name":"shart","price":"rs.30.00","colors":["red"] } ]

# result = itms.insert_many(item2)

def itemsPrice():
    s=''
    results = itms.find()
    for result in results:
        s= s + result["price"] + " ,"
    return s

def itemsColors():
    s=''
    results = itms.find()
    for result in results:
        for color in result["colors"]:
            s = s + color + " ,"
    return s
