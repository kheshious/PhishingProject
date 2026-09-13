import html
import os
import re
import smtplib

from datetime import datetime
from email.message import EmailMessage


URL_PATTERN = re.compile(
    r"""
    (
        https?://[^\s<>"']+
        |
        www\.[^\s<>"']+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def create_report_reference(
    report_id,
    report_date=None,
):

    if isinstance(
        report_date,
        datetime
    ):

        report_year = (
            report_date.year
        )

    else:

        report_year = (
            datetime.now().year
        )

    return (
        f"SCAM-{report_year}-"
        f"{int(report_id):04d}"
    )


def create_safe_content_preview(
    reported_content,
    maximum_length=180,
):

    content = str(
        reported_content or ""
    ).strip()

    content = URL_PATTERN.sub(
        "[link removed]",
        content
    )

    content = re.sub(
        r"\s+",
        " ",
        content
    ).strip()

    if not content:

        return (
            "No content preview is available."
        )

    if len(content) > maximum_length:

        content = (
            content[
                :maximum_length
            ].rstrip()
            + "..."
        )

    return content


def get_safety_guidance(
    report_status,
):

    guidance = {
        "Confirmed Scam": (
            "Do not respond to the message, open attachments, "
            "follow links or provide personal information. "
            "Block the sender where appropriate."
        ),

        "Not a Scam": (
            "The report was not confirmed as a scam. "
            "Continue to verify unexpected requests before "
            "providing sensitive information."
        ),

        "Reviewed": (
            "Please read the administrator feedback carefully "
            "and continue to follow normal online safety precautions."
        ),
    }

    return guidance.get(
        report_status,
        (
            "Continue to exercise caution and verify unexpected "
            "requests through an official communication channel."
        )
    )


def send_report_review_email(
    report,
):

    smtp_host = os.environ.get(
        "SMTP_HOST",
        ""
    ).strip()

    smtp_port = int(
        os.environ.get(
            "SMTP_PORT",
            "587"
        )
    )

    smtp_username = os.environ.get(
        "SMTP_USERNAME",
        ""
    ).strip()

    smtp_password = os.environ.get(
        "SMTP_PASSWORD",
        ""
    )

    sender_email = os.environ.get(
        "SMTP_FROM_EMAIL",
        smtp_username
    ).strip()

    sender_name = os.environ.get(
        "SMTP_FROM_NAME",
        "PhishAI"
    ).strip()

    recipient_email = str(
        report.get(
            "email",
            ""
        )
        or ""
    ).strip()

    if not smtp_host:

        raise ValueError(
            "SMTP_HOST has not been configured."
        )

    if not smtp_username:

        raise ValueError(
            "SMTP_USERNAME has not been configured."
        )

    if not smtp_password:

        raise ValueError(
            "SMTP_PASSWORD has not been configured."
        )

    if not sender_email:

        raise ValueError(
            "SMTP_FROM_EMAIL has not been configured."
        )

    if not recipient_email:

        raise ValueError(
            "The reporting user does not have an email address."
        )

    report_reference = create_report_reference(
        report_id=report["report_id"],
        report_date=report.get(
            "report_date"
        ),
    )

    report_type = str(
        report.get(
            "report_type",
            "Unknown"
        )
    )

    report_status = str(
        report.get(
            "status",
            "Reviewed"
        )
    )

    admin_feedback = str(
        report.get(
            "admin_feedback",
            ""
        )
        or (
            "The report was reviewed, but no additional "
            "feedback was provided."
        )
    ).strip()

    first_name = str(
        report.get(
            "first_name",
            "User"
        )
        or "User"
    ).strip()

    safe_preview = (
        create_safe_content_preview(
            report.get(
                "reported_content",
                ""
            )
        )
    )

    safety_guidance = (
        get_safety_guidance(
            report_status
        )
    )

    subject = (
        f"PhishAI report update: "
        f"{report_reference}"
    )

    plain_text_body = f"""
Hello {first_name},

Your reported item has been reviewed by a PhishAI administrator.

Report reference: {report_reference}
Report type: {report_type}
Review status: {report_status}

Submitted content preview:
{safe_preview}

Administrator feedback:
{admin_feedback}

Recommended action:
{safety_guidance}

For security reasons, links contained in the submitted content are removed from this email. You can sign in to PhishAI to view the complete report.

Regards,
PhishAI Security Awareness Platform
""".strip()

    html_body = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
</head>

<body
    style="
        margin: 0;
        padding: 0;
        background: #eef3f8;
        font-family: Arial, Helvetica, sans-serif;
        color: #172033;
    "
>

    <div
        style="
            max-width: 640px;
            margin: 30px auto;
            padding: 0 16px;
        "
    >

        <div
            style="
                background: linear-gradient(
                    135deg,
                    #1769c7,
                    #0d315f
                );
                border-radius: 14px 14px 0 0;
                padding: 26px 30px;
                color: #ffffff;
            "
        >

            <h1
                style="
                    margin: 0;
                    font-size: 25px;
                "
            >
                🛡️ PhishAI
            </h1>

            <p
                style="
                    margin: 8px 0 0;
                    color: #dbeafe;
                "
            >
                Scam Report Review
            </p>

        </div>


        <div
            style="
                background: #ffffff;
                border-radius: 0 0 14px 14px;
                padding: 30px;
                box-shadow:
                    0 10px 30px
                    rgba(15, 23, 42, 0.10);
            "
        >

            <p>
                Hello
                <strong>
                    {html.escape(first_name)}
                </strong>,
            </p>

            <p>
                Your reported item has been reviewed
                by a PhishAI administrator.
            </p>


            <div
                style="
                    margin: 22px 0;
                    padding: 18px;
                    border: 1px solid #dbe3ee;
                    border-radius: 10px;
                    background: #f8fafc;
                "
            >

                <p style="margin: 0 0 10px;">

                    <strong>
                        Report reference:
                    </strong>

                    {html.escape(report_reference)}

                </p>

                <p style="margin: 0 0 10px;">

                    <strong>
                        Report type:
                    </strong>

                    {html.escape(report_type)}

                </p>

                <p style="margin: 0;">

                    <strong>
                        Review status:
                    </strong>

                    {html.escape(report_status)}

                </p>

            </div>


            <h3 style="color: #0f4fa3;">
                Submitted content preview
            </h3>

            <div
                style="
                    border-left: 4px solid #1769c7;
                    border-radius: 7px;
                    background: #eff6ff;
                    padding: 14px 16px;
                    line-height: 1.6;
                "
            >
                {html.escape(safe_preview)}
            </div>


            <h3 style="color: #0f4fa3;">
                Administrator feedback
            </h3>

            <div
                style="
                    border-left: 4px solid #16a34a;
                    border-radius: 7px;
                    background: #f0fdf4;
                    padding: 14px 16px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                "
            >
                {html.escape(admin_feedback)}
            </div>


            <h3 style="color: #0f4fa3;">
                Recommended action
            </h3>

            <p style="line-height: 1.6;">
                {html.escape(safety_guidance)}
            </p>


            <p
                style="
                    margin-top: 25px;
                    padding-top: 18px;
                    border-top: 1px solid #e2e8f0;
                    color: #64748b;
                    font-size: 13px;
                    line-height: 1.5;
                "
            >
                For security reasons, links contained in the
                submitted content are removed from this email.
                Sign in to PhishAI to view the complete report.
            </p>


            <p style="margin-bottom: 0;">
                Regards,<br>
                <strong>
                    PhishAI Security Awareness Platform
                </strong>
            </p>

        </div>

    </div>

</body>

</html>
""".strip()

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = (
        f"{sender_name} <{sender_email}>"
    )
    message["To"] = recipient_email

    message.set_content(
        plain_text_body
    )

    message.add_alternative(
        html_body,
        subtype="html"
    )

    if smtp_port == 465:

        with smtplib.SMTP_SSL(
            smtp_host,
            smtp_port,
            timeout=30,
        ) as smtp_server:

            smtp_server.login(
                smtp_username,
                smtp_password,
            )

            smtp_server.send_message(
                message
            )

    else:

        with smtplib.SMTP(
            smtp_host,
            smtp_port,
            timeout=30,
        ) as smtp_server:

            smtp_server.ehlo()

            smtp_server.starttls()

            smtp_server.ehlo()

            smtp_server.login(
                smtp_username,
                smtp_password,
            )

            smtp_server.send_message(
                message
            )

    return {
        "sent": True,
        "recipient": recipient_email,
        "report_reference": report_reference,
    }