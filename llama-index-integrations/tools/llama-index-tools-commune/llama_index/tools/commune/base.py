"""Commune email and SMS tool spec for LlamaIndex agents."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class CommuneToolSpec(BaseToolSpec):
    """Commune email and SMS tools for LlamaIndex agents.

    Commune (https://commune.email) provides email and SMS infrastructure
    designed for AI agents. This ToolSpec wraps the commune-mail Python SDK,
    giving agents a real inbox, the ability to send and receive messages, and
    structured output the LLM can act on.

    To use, set the COMMUNE_API_KEY environment variable or pass api_key
    directly. Obtain a key at https://commune.email.

    Example:
        .. code-block:: python

            import os
            from llama_index.tools.commune import CommuneToolSpec

            tools = CommuneToolSpec(api_key=os.environ["COMMUNE_API_KEY"])
            agent = ReActAgent.from_tools(tools.to_tool_list(), llm=llm)
    """

    spec_functions = [
        "load_inbox",
        "search_emails",
        "get_email",
        "send_email",
        "send_sms",
        "get_credits",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize CommuneToolSpec.

        Args:
            api_key: Commune API key. Falls back to the COMMUNE_API_KEY
                environment variable if not provided.

        Raises:
            ValueError: If no API key is found.
        """
        resolved_key = api_key or os.environ.get("COMMUNE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Commune API key is required. Pass api_key= or set the "
                "COMMUNE_API_KEY environment variable. Get a key at "
                "https://commune.email."
            )
        try:
            from commune import Commune  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "commune-mail is required. Install it with: "
                "pip install commune-mail"
            ) from exc

        self._client = Commune(api_key=resolved_key)
        super().__init__()

    # ------------------------------------------------------------------
    # Email tools
    # ------------------------------------------------------------------

    def load_inbox(
        self,
        limit: int = 20,
        unread_only: bool = False,
    ) -> str:
        """Fetch recent emails from the inbox.

        Use this tool when the agent needs to check for new messages, review
        pending requests, or get an overview of recent email activity. It
        returns a structured list of emails with sender, subject, timestamp,
        and a preview of the body.

        Prefer unread_only=True for a focused view of messages that have not
        been processed yet. Use a higher limit (up to 50) when the agent
        needs broader context about recent conversations.

        Args:
            limit: Maximum number of emails to return (default: 20, max: 50).
            unread_only: If True, only return emails that have not been read
                yet. Default is False (returns all recent emails).

        Returns:
            A formatted string listing each email with index, ID, sender,
            subject, date, read status, and a short body preview. If the
            inbox is empty, returns a message indicating no emails were found.

        Example output::

            Found 2 email(s):

            [1] ID: msg_abc123
                From: alice@example.com
                Subject: Project update
                Date: 2024-01-15T10:30:00Z
                Read: Yes
                Preview: Hi, just wanted to share the latest status...

            [2] ID: msg_def456
                From: bob@example.com
                Subject: Invoice #1042
                Date: 2024-01-14T08:00:00Z
                Read: No
                Preview: Please find attached the invoice for January...
        """
        try:
            emails: List[Any] = self._client.emails.list(
                limit=limit, unread_only=unread_only
            )
        except Exception as exc:
            return f"Error fetching inbox: {exc}"

        if not emails:
            label = "unread " if unread_only else ""
            return f"No {label}emails found in the inbox."

        lines = [f"Found {len(emails)} email(s):\n"]
        for idx, email in enumerate(emails, start=1):
            body_preview = self._preview(self._attr(email, "body"), 120)
            lines.append(
                f"[{idx}] ID: {self._attr(email, 'id')}\n"
                f"    From: {self._attr(email, 'from')}\n"
                f"    Subject: {self._attr(email, 'subject')}\n"
                f"    Date: {self._attr(email, 'received_at')}\n"
                f"    Read: {'Yes' if self._attr(email, 'read') else 'No'}\n"
                f"    Preview: {body_preview}\n"
            )
        return "\n".join(lines)

    def search_emails(
        self,
        query: str,
        limit: int = 10,
    ) -> str:
        """Search emails by keyword, topic, sender, or subject.

        Use this tool when you need to find specific emails — e.g., locate
        all messages about a particular invoice, find emails from a specific
        sender, or retrieve conversation history on a topic. This is more
        precise than load_inbox when you know what you're looking for.

        The search operates across subject lines, sender addresses, and body
        content. Use natural language queries or specific identifiers.

        Args:
            query: The search query string. Can be a keyword ("invoice"),
                a sender address ("alice@example.com"), a topic
                ("deployment status"), or any combination.
            limit: Maximum number of results to return (default: 10).

        Returns:
            A formatted string listing matching emails. Each entry includes
            the email ID, sender, subject, date, and a body preview. If no
            emails match, returns a message saying so.

        Example output::

            Found 1 email(s) matching "invoice":

            [1] ID: msg_def456
                From: billing@vendor.com
                Subject: Invoice #1042 for January
                Date: 2024-01-14T08:00:00Z
                Preview: Please find the attached invoice totalling $2,400...
        """
        try:
            results: List[Any] = self._client.emails.search(
                query=query, limit=limit
            )
        except Exception as exc:
            return f"Error searching emails for '{query}': {exc}"

        if not results:
            return f"No emails found matching '{query}'."

        lines = [f"Found {len(results)} email(s) matching \"{query}\":\n"]
        for idx, email in enumerate(results, start=1):
            body_preview = self._preview(self._attr(email, "body"), 120)
            lines.append(
                f"[{idx}] ID: {self._attr(email, 'id')}\n"
                f"    From: {self._attr(email, 'from')}\n"
                f"    Subject: {self._attr(email, 'subject')}\n"
                f"    Date: {self._attr(email, 'received_at')}\n"
                f"    Preview: {body_preview}\n"
            )
        return "\n".join(lines)

    def get_email(self, email_id: str) -> str:
        """Retrieve the full content of a specific email by its ID.

        Use this tool after load_inbox or search_emails when you need to read
        the complete body of a particular email. The inbox and search tools
        return only a short preview; use this tool when the agent needs the
        full text to understand, summarise, or respond to a message.

        Args:
            email_id: The unique email identifier returned by load_inbox or
                search_emails (e.g., "msg_abc123").

        Returns:
            A formatted string containing the full email details: ID, sender,
            recipient(s), subject, date, read status, and the complete body.

        Example output::

            Email ID: msg_abc123
            From: alice@example.com
            To: agent@myapp.commune.email
            Subject: Project update
            Date: 2024-01-15T10:30:00Z
            Read: Yes

            Body:
            Hi, just wanted to share the latest project status. Everything is
            on track for the Friday deadline. The team completed the backend
            migration yesterday and QA starts tomorrow...
        """
        if not email_id or not email_id.strip():
            return "Error: email_id is required and cannot be empty."

        try:
            email = self._client.emails.get(email_id=email_id.strip())
        except Exception as exc:
            return f"Error retrieving email '{email_id}': {exc}"

        if email is None:
            return f"No email found with ID '{email_id}'."

        return (
            f"Email ID: {self._attr(email, 'id')}\n"
            f"From: {self._attr(email, 'from')}\n"
            f"To: {self._attr(email, 'to')}\n"
            f"Subject: {self._attr(email, 'subject')}\n"
            f"Date: {self._attr(email, 'received_at')}\n"
            f"Read: {'Yes' if self._attr(email, 'read') else 'No'}\n"
            f"\nBody:\n{self._attr(email, 'body')}"
        )

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ) -> str:
        """Compose and send an email to one or more recipients.

        Use this tool when the agent needs to reply to a user, send a
        notification, deliver a report, or communicate with another agent or
        service via email. The body should be written in plain text.

        If from_address is not provided, the default sender address configured
        for your Commune account will be used (typically something like
        agent@yourapp.commune.email).

        Args:
            to: Recipient email address (e.g., "user@example.com"). To send
                to multiple recipients, separate addresses with commas:
                "alice@example.com, bob@example.com".
            subject: The email subject line. Keep it concise and descriptive.
            body: The plain text body of the email. Write naturally; Commune
                handles formatting and delivery.
            from_address: Optional sender address. If omitted, the account
                default is used. Use this to send from a specific inbox, e.g.,
                "support@myapp.commune.email".

        Returns:
            A confirmation string with the message ID and delivery status if
            successful, or an error message if the send failed.

        Example output::

            Email sent successfully.
            Message ID: msg_xyz789
            Status: sent
            To: user@example.com
            Subject: Your request has been processed
        """
        if not to or not subject or not body:
            return (
                "Error: 'to', 'subject', and 'body' are all required to "
                "send an email."
            )

        kwargs: Dict[str, Any] = {
            "to": to.strip(),
            "subject": subject.strip(),
            "body": body.strip(),
        }
        if from_address:
            kwargs["from_address"] = from_address.strip()

        try:
            result = self._client.emails.send(**kwargs)
        except Exception as exc:
            return f"Error sending email to '{to}': {exc}"

        msg_id = self._attr(result, "id")
        status = self._attr(result, "status")
        return (
            f"Email sent successfully.\n"
            f"Message ID: {msg_id}\n"
            f"Status: {status}\n"
            f"To: {to}\n"
            f"Subject: {subject}"
        )

    # ------------------------------------------------------------------
    # SMS tools
    # ------------------------------------------------------------------

    def send_sms(self, to: str, body: str) -> str:
        """Send an SMS message to a phone number.

        Use this tool when the agent needs to send a short text message — for
        example, alerting a user to an urgent event, sending a verification
        code, delivering a status update, or notifying someone who has
        requested SMS communication.

        Phone numbers must be in E.164 format: a '+' followed by the country
        code and number with no spaces or dashes (e.g., "+15551234567" for a
        US number, "+447700900000" for a UK number).

        SMS messages are limited to 160 characters per segment. Longer
        messages are automatically split into multiple segments, which may
        use additional API credits.

        Args:
            to: Recipient phone number in E.164 format (e.g., "+15551234567").
            body: The text content of the SMS message.

        Returns:
            A confirmation string with the message ID and delivery status if
            successful, or an error message if the send failed.

        Example output::

            SMS sent successfully.
            Message ID: sms_abc123
            Status: sent
            To: +15551234567
        """
        if not to or not body:
            return "Error: 'to' and 'body' are required to send an SMS."

        if not to.strip().startswith("+"):
            return (
                "Error: Phone number must be in E.164 format starting with "
                "'+' followed by country code and number (e.g., +15551234567)."
            )

        try:
            result = self._client.sms.send(
                to=to.strip(), body=body.strip()
            )
        except Exception as exc:
            return f"Error sending SMS to '{to}': {exc}"

        msg_id = self._attr(result, "id")
        status = self._attr(result, "status")
        return (
            f"SMS sent successfully.\n"
            f"Message ID: {msg_id}\n"
            f"Status: {status}\n"
            f"To: {to}"
        )

    # ------------------------------------------------------------------
    # Account tools
    # ------------------------------------------------------------------

    def get_credits(self) -> str:
        """Check the current Commune API credit balance.

        Use this tool when the agent needs to verify that sufficient credits
        are available before performing a bulk operation (e.g., sending many
        emails), or when reporting on account status. This tool makes a
        lightweight API call and does not consume credits itself.

        Returns:
            A string reporting the current balance and currency. If the
            balance is low (below 5.00), the response includes a warning
            suggesting the user top up at https://commune.email.

        Example output::

            Commune API Credits
            Balance: $42.50 USD

        Low balance example::

            Commune API Credits
            Balance: $1.20 USD
            Warning: Credit balance is low. Top up at https://commune.email
                     to avoid service interruptions.
        """
        try:
            info = self._client.credits.get()
        except Exception as exc:
            return f"Error retrieving credit balance: {exc}"

        balance = self._attr(info, "balance")
        currency = self._attr(info, "currency") or "USD"

        try:
            numeric_balance = float(balance)
        except (TypeError, ValueError):
            numeric_balance = None

        output = f"Commune API Credits\nBalance: ${balance} {currency}"

        if numeric_balance is not None and numeric_balance < 5.0:
            output += (
                "\nWarning: Credit balance is low. Top up at "
                "https://commune.email to avoid service interruptions."
            )
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attr(obj: Any, key: str, default: str = "N/A") -> Any:
        """Safely retrieve an attribute from a dict or object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _preview(text: Any, length: int = 120) -> str:
        """Return a truncated preview of a string."""
        if not text or not isinstance(text, str):
            return "(no body)"
        text = text.strip().replace("\n", " ")
        if len(text) <= length:
            return text
        return text[:length].rstrip() + "..."
