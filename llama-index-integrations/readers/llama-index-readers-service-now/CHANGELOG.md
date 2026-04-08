# CHANGELOG

## [0.4.0] - 2026-04-05

### Added

- Support for OAuth2 Client Credentials Grant Flow authentication (machine-to-machine, no username/password required)
- Requires pysnc>=1.2.1 which includes `ServiceNowClientCredentialsFlow`

### Changed

- Authentication validation now supports three modes: basic auth, password grant flow, and client credentials flow
- `username` and `password` parameters are no longer always required (only needed for basic/password grant flow)
- Bumped minimum `pysnc` dependency from `>=1.0.0` to `>=1.2.1`

## [0.1.0] - 2025-07-15

### Added

- Initial release of ServiceNow Knowledge Base Reader
- Load KB articles by sys_id or KB numbers from ServiceNow instances
- OAuth2 password grant flow authentication
- Automatic attachment download and processing for multiple file types
- Custom parser system and event monitoring capabilities
- Configurable KB table and workflow state filtering
