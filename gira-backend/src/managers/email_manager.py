from django.core.mail import EmailMultiAlternatives, get_connection
from django.template.loader import render_to_string
from django.conf import settings

from gira_emails.models import GiraEmail, EmailStatus
from src.constants.email_constants import EmailType


class EmailManager:

    def __init__(self):
        self.connection = get_connection(
            backend="django.core.mail.backends.smtp.EmailBackend",
            host=settings.EMAIL_HOST,
            port=settings.EMAIL_PORT,
            username=settings.EMAIL_HOST_USER,
            password=settings.EMAIL_HOST_PASSWORD,
            use_tls=settings.EMAIL_USE_TLS,
            use_ssl=settings.EMAIL_USE_SSL,
        )

    def send_email(
        self,
        subject: str,
        template_name: str,
        context: dict,
        to_email: str,
        email_type: EmailType,
    ):

        html_content = render_to_string(template_name, context)

        email_record = GiraEmail.objects.create(
            sender_email=settings.DEFAULT_FROM_EMAIL,
            receiver_email=to_email,
            email_type=email_type,
        )

        try:
            msg = EmailMultiAlternatives(
                subject=subject,
                body="",
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[to_email],
                connection=self.connection,
            )
            msg.attach_alternative(html_content, "text/html")

            sent_count = msg.send()
            email_record.status = (
                EmailStatus.SUCCESS if sent_count > 0 else EmailStatus.FAILED
            )

        except Exception as e:
            print(f"SMTP error while sending email: {e}")
            email_record.status = EmailStatus.FAILED

        finally:
            email_record.save()
            return email_record
