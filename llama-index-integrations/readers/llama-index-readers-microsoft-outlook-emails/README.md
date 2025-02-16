# Microsoft Outlook Email Reader

```bash
pip install llama-index-readers-microsoft-outlook-emails
```

The loader retrieves emails from an Outlook mailbox and indexes the subject and body of the emails.

## Prerequisites

### App Authentication using Microsoft Entra ID (formerly Azure AD)

1. You need to create an App Registration in Microsoft Entra ID. Refer [here](https://learn.microsoft.com/en-us/azure/healthcare-apis/register-application)
2. API Permissions for the created app:
   1. Microsoft Graph --> Application Permissions --> Mail.Read (**Grant Admin Consent**)

More info on Microsoft Graph APIs - [Refer here](https://learn.microsoft.com/en-us/graph/permissions-reference)

## Usage

To use this loader, `client_id`, `client_secret`, and `tenant_id` of the registered app in Microsoft Azure Portal are required.

This loader fetches emails from a specified folder in an Outlook mailbox.

```python
from llama_index.readers.outlook_emails import OutlookEmailReader

loader = OutlookEmailReader(
    client_id="<Client ID of the app>",
    client_secret="<Client Secret of the app>",
    tenant_id="<Tenant ID of the Microsoft Azure Directory>",
    user_email="<User Email Address>",
    folder="Inbox",
    num_mails=10,
)

documents = loader.load_data()
```

The loader retrieves the subject and body of the emails from the specified folder in Outlook.
