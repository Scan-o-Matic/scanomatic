__author__ = 'martin'


import scanomatic.io.logger as logger

import smtplib
import socket

try:
    from email import MIMEText, MIMEMultipart
except ImportError:
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText



_logger = logger.Logger("Mailer")

def get_server(host=None, smtp_port=0, tls=False, login=None, password=None):

    if host:
        if tls and smtp_port == 0:
            smtp_port = 587
        elif smtp_port == 0:
            smtp_port = 25

        try:
            server = smtplib.SMTP(host, port=smtp_port)
        except:
            return None

    else:
        try:
            server = smtplib.SMTP()
        except:
            return None

    if tls:

        server.ehlo()
        server.starttls()
        server.ehlo()

    if login:

        server.login(login, password)
    else:
        server.connect()

    return server


def mail(sender, receiver, subject, message, final_message=True, server=None):
    """

    :param sender: Then mail address of the sender, if has value `None` a default address will be generated using
    `get_default_email()`
    :type sender: str or None
    :param receiver: The mail address(es) of the reciever(s)
    :type receiver: str or [str]
    :param subject: Subject line
    :type subject: str
    :param message: Bulk of message
    :type message: str
    :param final_message (optional): If this is the final message intended to be sent by the server.
    If so, server will be disconnected afterwards. Default `True`
    :type final_message: bool
    :param server (optional): The server to send the message, if not supplied will create a default server
     using `get_server()`
    :type server: smtplib.SMTP
    :return: None
    """
    if server is None:
        server = get_server()

    if server is None:
        return

    if not sender:
        sender = get_default_email()

    try:
        msg = MIMEMultipart()
    except TypeError:
        msg = MIMEMultipart.MIMEMultipart()

    msg['From'] = sender
    msg['To'] = receiver if isinstance(receiver, str) else ", ".join(receiver)
    msg['Subject'] = subject
    try:
        msg.attach(MIMEText(message))
    except TypeError:
        msg.attach(MIMEText.MIMEText(message))

    if isinstance(receiver, str):
        receiver = [receiver]
    try:
        server.sendmail(sender, receiver, msg.as_string())
    except smtplib.SMTPException:
        _logger.error("Could not mail, either no network connection or missing mailing functionality.")

    if final_message:
        try:
            server.quit()
        except:
            pass


def get_host_name():

    try:
        ip = [(s.connect(('8.8.8.8', 80)), s.getsockname()[0], s.close())
              for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except IndexError:
        return None

    return socket.gethostbyaddr(ip)[0]


def get_default_email(username="no-reply---scan-o-matic"):

    hostname = get_host_name()
    if not hostname:
        hostname = "scanomatic.somewhere"

    return "{0}@{1}".format(username, hostname)