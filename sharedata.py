import pymongo
from pymongo import MongoClient
from args import processed_data_path
import pandas as pd
if __name__ == "__main__":
    client = MongoClient(
        "mongodb+srv://Patrick:WangyunLuo@clustertest-nwszm.mongodb.net/test?retryWrites=true&w=majority")
    db = client["my-efts-app"]
    collection = db["my-etfs-app"]
    data = pd.read_csv(processed_data_path)
    data_dict = data.to_dict(orient='record')
    collection.insert_many(data_dict)

