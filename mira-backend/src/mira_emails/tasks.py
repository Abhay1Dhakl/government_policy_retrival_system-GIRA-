import os

from celery import shared_task

from src.constants.email_constants import EmailType
from src.managers.email_manager import EmailManager


@shared_task
def send_invite_mail(user_email, first_name):
    try:
        invite_link = (
            f"{os.getenv('FRONTEND_BASE_URL', 'http://localhost:3000')}/create-password"
        )
        subject = "You're Invited to Join Mira!"
        context = {"first_name": first_name, "invite_link": invite_link}

        email_manager = EmailManager()
        email_manager.send_email(
            subject=subject,
            template_name="mira_emails/invite_user.html",
            context=context,
            to_email=user_email,
            email_type=EmailType.INVITE_NEW_USER,
        )

    except Exception as e:
        print(f"Error preparing invite email: {e}")
        raise e
