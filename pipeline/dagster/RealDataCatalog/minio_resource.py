from dagster import ConfigurableResource
from minio import Minio

class MinioResource(ConfigurableResource):

    endpoint: str
    access_key: str
    secret_key: str
    port: str
    bucket_name: str

    def get_client(self):
        client = Minio(self.endpoint + ":" + self.port, self.access_key, self.secret_key, secure=False)
        found = client.bucket_exists(self.bucket_name)
        if not found:
            client.make_bucket(self.bucket_name)
        return client

    def get_object_presigned_url(self, object_name: str):
        client = self.get_client()
        return client.presigned_get_object(self.bucket_name, object_name)

    def list_objects(self, prefix: str) -> list:
        client = self.get_client()
        return list(client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
    
    def put_object(self, object_name: str, data, size: int, content_type: str):
        client = self.get_client()

        return client.put_object(self.bucket_name, object_name, data, size, content_type)

    def get_object(self, object_name: str):
        client = self.get_client()
        return client.get_object(self.bucket_name, object_name)