from datetime import datetime
import boto3
from storage.settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_BUCKET, AWS_REGION_NAME


class Boto3Client:
    def __init__(self):
        self.client = boto3.client(
            service_name='s3',
            region_name=AWS_REGION_NAME,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    def upload(self, file, file_name):
        save_file_name = f'{datetime.now().timestamp()}-{file_name}'

        self.client.upload_fileobj(
            file,
            AWS_DEFAULT_BUCKET,
            f'input/{save_file_name}'
        )

        return save_file_name

    def download(self, file_name):
        # TODO memory에 저장
        self.client.download_fileobj(
            AWS_DEFAULT_BUCKET, f'input/{file_name}', file_name
        )

        return file_name
