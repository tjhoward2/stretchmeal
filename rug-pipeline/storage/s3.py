"""AWS S3 storage backend with Glacier Deep Archive lifecycle support.

Implements the StorageBackend protocol using boto3. Supports automatic
Glacier Deep Archive transitions for the archive bucket.
"""

import json
import logging
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Backend:
    """Storage backend using AWS S3.

    Args:
        region: AWS region for the S3 client.
        glacier_transition_days: Days before archive objects transition
            to Glacier Deep Archive. Set to 0 to disable.
    """

    def __init__(self, region: str = "us-east-1", glacier_transition_days: int = 1):
        self._s3 = boto3.client("s3", region_name=region)
        self._region = region
        self._glacier_transition_days = glacier_transition_days
        self._lifecycle_configured: set[str] = set()

    def put(self, local_path: Path, remote_key: str, bucket: str) -> None:
        """Upload a local file to S3."""
        self._ensure_bucket(bucket)
        logger.info("S3 put: %s → s3://%s/%s", local_path, bucket, remote_key)
        self._s3.upload_file(str(local_path), bucket, remote_key)

    def get(self, remote_key: str, bucket: str, local_path: Path) -> Path:
        """Download an S3 object to a local file."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("S3 get: s3://%s/%s → %s", bucket, remote_key, local_path)

        try:
            self._s3.download_file(bucket, remote_key, str(local_path))
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404" or error_code == "NoSuchKey":
                raise FileNotFoundError(
                    f"Key not found: s3://{bucket}/{remote_key}"
                ) from e
            if error_code == "InvalidObjectState":
                raise RuntimeError(
                    f"Object s3://{bucket}/{remote_key} is archived in Glacier. "
                    f"Initiate a restore before downloading."
                ) from e
            raise

        return local_path

    def exists(self, remote_key: str, bucket: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self._s3.head_object(Bucket=bucket, Key=remote_key)
            return True
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                return False
            raise

    def delete(self, remote_key: str, bucket: str) -> None:
        """Delete an object from S3."""
        logger.info("S3 delete: s3://%s/%s", bucket, remote_key)
        self._s3.delete_object(Bucket=bucket, Key=remote_key)

    def list_keys(self, prefix: str, bucket: str) -> list[str]:
        """List all keys under a prefix in S3."""
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return sorted(keys)

    def read_json(self, remote_key: str, bucket: str) -> dict:
        """Download and parse a JSON file from S3."""
        try:
            response = self._s3.get_object(Bucket=bucket, Key=remote_key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise FileNotFoundError(
                    f"Key not found: s3://{bucket}/{remote_key}"
                ) from e
            raise

    def write_json(self, data: dict, remote_key: str, bucket: str) -> None:
        """Write a dict as JSON to S3."""
        self._ensure_bucket(bucket)
        body = json.dumps(data, indent=2).encode("utf-8")
        logger.info("S3 write_json: s3://%s/%s", bucket, remote_key)
        self._s3.put_object(Bucket=bucket, Key=remote_key, Body=body)

    # --- Glacier lifecycle ---

    def configure_glacier_lifecycle(self, bucket: str) -> None:
        """Set up Glacier Deep Archive lifecycle rule on a bucket.

        Idempotent — checks if the rule already exists before creating it.
        """
        if self._glacier_transition_days <= 0:
            logger.info("Glacier transitions disabled (days=%d)", self._glacier_transition_days)
            return

        if bucket in self._lifecycle_configured:
            return

        rule_id = "rug-pipeline-glacier-archive"

        # Check if rule already exists
        try:
            response = self._s3.get_bucket_lifecycle_configuration(Bucket=bucket)
            for rule in response.get("Rules", []):
                if rule.get("ID") == rule_id:
                    logger.info("Glacier lifecycle rule already exists on bucket %s", bucket)
                    self._lifecycle_configured.add(bucket)
                    return
            existing_rules = response.get("Rules", [])
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                existing_rules = []
            else:
                raise

        # Add our rule alongside any existing rules
        new_rule = {
            "ID": rule_id,
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Transitions": [
                {
                    "Days": self._glacier_transition_days,
                    "StorageClass": "DEEP_ARCHIVE",
                }
            ],
        }

        all_rules = existing_rules + [new_rule]

        self._s3.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration={"Rules": all_rules},
        )

        logger.info(
            "Configured Glacier Deep Archive transition on bucket %s "
            "(transition after %d days)",
            bucket, self._glacier_transition_days,
        )
        self._lifecycle_configured.add(bucket)

    # --- Helpers ---

    def _ensure_bucket(self, bucket: str) -> None:
        """Create the S3 bucket if it doesn't exist."""
        try:
            self._s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                logger.info("Creating S3 bucket: %s in %s", bucket, self._region)
                if self._region == "us-east-1":
                    self._s3.create_bucket(Bucket=bucket)
                else:
                    self._s3.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={
                            "LocationConstraint": self._region
                        },
                    )
            else:
                raise
