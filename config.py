import json

class DatabaseConfig:
    def __init__(self, database_name="",
                 collection_name="",
                 dataset_dir=""):
        self.database_name = database_name
        self.collection_name = collection_name
        self.dataset_dir = dataset_dir

    @classmethod
    def from_dict(cls, json_object):
        config = DatabaseConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))
