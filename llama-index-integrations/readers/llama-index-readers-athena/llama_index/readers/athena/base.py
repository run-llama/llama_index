"""Athena Reader."""

import warnings
from typing import Optional

import boto3
from llama_index.core.readers.base import BaseReader
from sqlalchemy.engine import create_engine


class AthenaReader(BaseReader):
    """
    Athena reader.

    Follow AWS best practices for security.
    AWS discourages hardcoding credentials in code.
    We recommend that you use IAM roles instead of IAM user credentials.
    If you must use credentials, do not embed them in your code.
    Instead, store them in environment variables or in a separate configuration file.

    """

    def __init__(
        self,
    ) -> None:
        """Initialize with parameters."""

    def create_athena_engine(
        self,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: str = None,
        s3_staging_dir: str = None,
        database: str = None,
        workgroup: str = None,
    ):
        """
        Args:
        aws_access_key is the AWS access key from aws credential
        aws_secret_key is the AWS secret key from aws credential
        aws_region is the AWS region
        s3_staging_dir is the S3 staging (result bucket) directory
        database is the Athena database name
        workgroup is the Athena workgroup name.

        """
        if not aws_access_key or not aws_secret_key:
            conn_str = (
                "awsathena+rest://:@athena.{region_name}.amazonaws.com:443/"
                "{database}?s3_staging_dir={s3_staging_dir}?work_group={workgroup}"
            )

            engine = create_engine(
                conn_str.format(
                    region_name=aws_region,
                    s3_staging_dir=s3_staging_dir,
                    database=database,
                    workgroup=workgroup,
                )
            )

        else:
            warnings.warn(
                "aws_access_key and aws_secret_key are set. We recommend to use IAM role instead."
            )
            boto3.client(
                "athena",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )

            conn_str = (
                "awsathena+rest://:@athena.{region_name}.amazonaws.com:443/"
                "{database}?s3_staging_dir={s3_staging_dir}?work_group={workgroup}"
            )

            engine = create_engine(
                conn_str.format(
                    region_name=aws_region,
                    s3_staging_dir=s3_staging_dir,
                    database=database,
                    workgroup=workgroup,
                )
            )
        return engine
