import boto3
import pickle
from abc import ABC, abstractmethod
import logging


class Database(ABC):
    @abstractmethod
    def add_emb(self, id_, emb):
        ...

    @abstractmethod
    def update_emb(self, id_, emb):
        ...

    @abstractmethod
    def del_emb(self, id_):
        ...

    @abstractmethod
    def get_emb(self, id_):
        ...

    @abstractmethod
    def add_user(self, id_):
        ...

    @abstractmethod
    def del_user(self, id_):
        ...

    @abstractmethod
    def add_channel(self, id_, channel):
        ...

    @abstractmethod
    def del_channel(self, id_, channel):
        ...

    def __init__(self, config):
        self.config = config


class S3Database(Database):
    def __init__(self, config):
        super().__init__(config)
        self.S3_BUCKET = self.config['S3_BUCKET']
        self.SERVICE_NAME = self.config['SERVICE_NAME']
        self.KEY = self.config['KEY']
        self.SECRET = self.config['SECRET']
        self.ENDPOINT = self.config['ENDPOINT']
        self.SESSION = boto3.session.Session()
        self.client = self.SESSION.client(
            service_name=self.SERVICE_NAME,
            aws_access_key_id=self.KEY,
            aws_secret_access_key=self.SECRET,
            endpoint_url=self.ENDPOINT
        )
        self.resource = self.SESSION.resource(
            service_name=self.SERVICE_NAME,
            aws_access_key_id=self.KEY,
            aws_secret_access_key=self.SECRET,
            endpoint_url=self.ENDPOINT
        )

    def add_emb(self, id_, emb):
        try:
            emb = pickle.dumps(emb.cpu().numpy(), protocol=2)
            self.client.put_object(Bucket=self.S3_BUCKET, Key=id_, Body=emb, StorageClass='COLD')
            return True
        except Exception as e:
            logging.error(f'ML: {e}')
            return False

    def update_emb(self, id_, emb):
        try:
            emb = pickle.dumps(emb.cpu().numpy(), protocol=2)
            self.client.put_object(Bucket=self.S3_BUCKET, Key=id_, Body=emb, StorageClass='COLD')
            return True
        except Exception as e:
            logging.error(f'ML: {e}')
            return False

    def del_emb(self, id_):
        if self.check(id_):
            self.resource.Object(Bucket=self.S3_BUCKET, Key=id_).delete()
            return True
        else:
            return False

    def get_emb(self, id_):
        if self.check(id_):
            return pickle.loads(self.client.get_object(Bucket=self.S3_BUCKET, Key=id_)['Body'])
        else:
            return None

    def add_user(self, id_):
        return True

    def del_user(self, id_):
        return True

    def add_channel(self, id_, channel):
        return True

    def del_channel(self, id_, channel):
        return True

    def check(self, key):
        try:
            self.client.head_object(Bucket=self.S3_BUCKET, Key=key)
            return True
        except:
            logging.info('ML key not found')
            return False
