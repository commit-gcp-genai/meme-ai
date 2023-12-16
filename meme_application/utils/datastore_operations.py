from google.cloud import datastore
import datetime


class DatastoreManager:
    def __init__(self):
        self.client = datastore.Client()

    def create_entity(self, kind, name, properties):
        key = self.client.key(kind, name)
        entity = datastore.Entity(key)
        for prop, value in properties.items():
            entity[prop] = value
        entity["last_interaction"] = datetime.datetime.now()
        self.client.put(entity)

    def update_entity(self, kind, name, updates):
        key = self.client.key(kind, name)
        entity = self.client.get(key)
        for prop, value in updates.items():
            entity[prop] = value
        entity["last_interaction"] = datetime.datetime.now()
        self.client.put(entity)

    def get_entity(self, kind, name):
        key = self.client.key(kind, name)
        return self.client.get(key)

    def delete_entity(self, kind, name):
        key = self.client.key(kind, name)
        self.client.delete(key)

    def query_entities(self, kind):
        query = self.client.query(kind=kind)
        return list(query.fetch())
