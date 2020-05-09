import boto3
import logging
from botocore.exceptions import ClientError
from utilities.config import region_name
from utilities.aws_keys import aws_access_key_id, aws_secret_access_key

s3_client = boto3.client('s3', region_name=region_name,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)


def download_from_s3(bucket: str, key: str, dest_pathname: str):
    try:
        s3_client.download_file(bucket, key, dest_pathname)
        logging.info('Se han descargado los datos desde S3 correctamente.')
    except (Exception, ClientError) as e:
        logging.error(f'Error descargando desde S3, {e}')
        if e.response['Error']['Code'] == '404':
            logging.info('El objeto no existe')
    return True


def upload_to_s3(filename: str, bucket: str, key: str, with_kms: bool = False):
    if with_kms:
        try:
            s3_client.upload_file(filename, bucket, key, extra_args={'ServerSideEncryption:': 'aws:kms',
                                                                     'SSEKMSKeyId': '<<your_kms_key>>'})
            logging.info('Se han cargado los datos en S3 correctamente')
        except (Exception, ClientError) as e:
            logging.error(f'Error cargando a S3, {e}')
    else:
        try:
            s3_client.upload_file(filename, bucket, key)
            logging.info('Se han cargado los datos en S3 correctamente')
        except (Exception, ClientError) as e:
            logging.error(f'Error cargando a S3, {e}')
    return True
