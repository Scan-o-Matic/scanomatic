__author__ = 'martin'


import smtplib

try:
    from email import MIMEText, MIMEMultipart
except ImportError:
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText


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

    if server is None:
        server = get_server('localhost')

    if server is None:
        return
    try:
        msg = MIMEMultipart()
    except TypeError:
        msg = MIMEMultipart.MIMEMultipart()

    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(message))

    server.sendmail(sender, [receiver], msg.as_string())

    if final_message:
        try:
            server.quit()
        except:
            pass