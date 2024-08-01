from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import botocore


def get_aws_service_client(
    service_name: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    profile_name: Optional[str] = None,
    max_retries: Optional[int] = 3,
    timeout: Optional[float] = 60.0,
) -> "botocore.client.BaseClient":
    try:
        import boto3
        import botocore
    except ImportError:
        raise ImportError(
            "Please run `pip install boto3 botocore` to use AWS services."
        )

    config = botocore.config.Config(
        retries={"max_attempts": max_retries, "mode": "standard"},
        connect_timeout=timeout,
    )

    try:
        if not profile_name and aws_access_key_id:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            client = session.client(service_name, config=config)
        else:
            session = boto3.Session(profile_name=profile_name)
            if region_name:
                client = session.client(
                    service_name, region_name=region_name, config=config
                )
            else:
                client = session.client(service_name, config=config)
    except Exception as e:
        raise ValueError("Please verify the provided credentials.") from (e)

    return client
