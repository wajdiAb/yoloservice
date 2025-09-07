# s3_utils.py
import os
import mimetypes
import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

_session = None
_s3 = None


def get_s3_client():
    global _session, _s3
    if _s3 is not None:
        return _s3
    _session = boto3.session.Session(region_name=AWS_REGION or None)
    _s3 = _session.client("s3")
    return _s3


def s3_key_exists(bucket: str, key: str) -> bool:
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_file(bucket: str, key: str, local_path: str) -> None:
    get_s3_client().download_file(bucket, key, local_path)


def upload_file(bucket: str, key: str, local_path: str, content_type: str | None = None) -> None:
    if not content_type:
        guessed, _ = mimetypes.guess_type(local_path)
        content_type = guessed or "application/octet-stream"
    get_s3_client().upload_file(
        local_path, bucket, key, ExtraArgs={"ContentType": content_type}
    )


def copy_object(bucket: str, src_key: str, dst_key: str) -> None:
    get_s3_client().copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        MetadataDirective="COPY",
    )
