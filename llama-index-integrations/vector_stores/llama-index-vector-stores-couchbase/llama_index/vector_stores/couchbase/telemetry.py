from importlib.metadata import version

try:
    from reo_census import ReoEventLogger

    _pkg_version = version("llama-index-vector-stores-couchbase")
    _logger = ReoEventLogger(
        endpoint_url="https://telemetry.reo.dev/data",
        timeout=3.0,
        package_name="llama-index-vector-stores-couchbase",
        package_version=_pkg_version,
    )
    _logger.log_event({"activity_type": "package_import"})
except Exception:
    pass
