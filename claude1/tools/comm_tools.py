"""Communication tools: email and webhooks."""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from claude1.tools.base import BaseTool

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False


class SendEmailTool(BaseTool):
    """Send an email via SMTP."""

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return (
            "Send an email via SMTP. Requires SMTP_HOST, SMTP_PORT, SMTP_USER, "
            "SMTP_PASSWORD, and SMTP_FROM environment variables to be set. "
            "Supports plain text and HTML bodies. "
            "Requires confirmation before sending."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address (comma-separated for multiple)",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Plain text email body",
                },
                "html_body": {
                    "type": "string",
                    "description": "Optional HTML email body (sent as alternative to plain text)",
                },
                "cc": {
                    "type": "string",
                    "description": "CC recipients (comma-separated)",
                },
            },
            "required": ["to", "subject", "body"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        to = kwargs.get("to", "")
        subject = kwargs.get("subject", "")
        body = kwargs.get("body", "")
        html_body = kwargs.get("html_body")
        cc = kwargs.get("cc", "")

        if not to or not subject or not body:
            return "Error: to, subject, and body are required"

        # Read SMTP config from environment
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_port_str = os.environ.get("SMTP_PORT", "587")
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_password = os.environ.get("SMTP_PASSWORD", "")
        smtp_from = os.environ.get("SMTP_FROM", "")

        if not smtp_host:
            return "Error: SMTP_HOST environment variable is not set. Configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM."

        try:
            smtp_port = int(smtp_port_str)
        except ValueError:
            return f"Error: Invalid SMTP_PORT value: {smtp_port_str}"

        sender = smtp_from or smtp_user
        if not sender:
            return "Error: SMTP_FROM or SMTP_USER must be set"

        # Build message
        recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
        cc_list = [addr.strip() for addr in cc.split(",") if addr.strip()] if cc else []

        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)

        msg.attach(MIMEText(body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        all_recipients = recipients + cc_list

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                if smtp_port == 587:
                    server.starttls()
                    server.ehlo()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(sender, all_recipients, msg.as_string())

            return f"Email sent successfully to {', '.join(recipients)}" + (f" (cc: {', '.join(cc_list)})" if cc_list else "")

        except smtplib.SMTPAuthenticationError:
            return "Error: SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD."
        except smtplib.SMTPConnectError as e:
            return f"Error: Could not connect to SMTP server {smtp_host}:{smtp_port}: {e}"
        except smtplib.SMTPException as e:
            return f"Error sending email: {e}"
        except Exception as e:
            return f"Error: {e}"


class WebhookTool(BaseTool):
    """POST JSON to a webhook URL."""

    @property
    def name(self) -> str:
        return "send_webhook"

    @property
    def description(self) -> str:
        return (
            "Send a JSON payload to a webhook URL via HTTP POST. "
            "Works with Slack, Discord, custom webhooks, and any service that accepts JSON POSTs. "
            "For Slack: pass message as the 'text' field. "
            "Requires confirmation before sending."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "webhook_url": {
                    "type": "string",
                    "description": "The webhook URL to POST to",
                },
                "message": {
                    "type": "string",
                    "description": "Simple text message (sent as {\"text\": message} for Slack-compatible webhooks)",
                },
                "payload": {
                    "type": "object",
                    "description": "Custom JSON payload (overrides message if both provided)",
                },
            },
            "required": ["webhook_url"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        if not HTTPX_AVAILABLE:
            return "Error: httpx is not installed. Run: pip install httpx"

        webhook_url = kwargs.get("webhook_url", "")
        message = kwargs.get("message", "")
        payload = kwargs.get("payload")

        if not webhook_url:
            return "Error: webhook_url is required"
        if not message and not payload:
            return "Error: Either message or payload is required"

        # Build JSON payload
        if payload:
            json_body = payload
        else:
            json_body = {"text": message}

        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.post(
                    webhook_url,
                    json=json_body,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code < 300:
                return f"Webhook sent successfully (status {response.status_code})"
            else:
                return f"Webhook returned status {response.status_code}: {response.text[:500]}"

        except httpx.TimeoutException:
            return f"Error: Webhook request timed out for {webhook_url}"
        except httpx.RequestError as e:
            return f"Error sending webhook: {e}"
        except Exception as e:
            return f"Error: {e}"
