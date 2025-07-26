from dagster import Definitions, load_assets_from_modules, EnvVar

from RealDataCatalog import assets
from RealDataCatalog.minio_resource import MinioResource

all_assets = load_assets_from_modules([assets])

resources = {
    "minio": MinioResource(
        endpoint=EnvVar("MINIO_ENDPOINT"),
        access_key=EnvVar("MINIO_ACCESS_KEY"),
        secret_key=EnvVar("MINIO_SECRET_KEY"),
        port=EnvVar("MINIO_PORT"),
        bucket_name=EnvVar("MINIO_BUCKET_NAME"),        
    )
}
defs = Definitions(
    assets=all_assets,
    resources=resources,
)
