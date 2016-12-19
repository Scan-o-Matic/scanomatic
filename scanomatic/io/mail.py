import scanomatic.io.logger as logger
from scanomatic.io.app_config import Config as AppConfig
from types import StringTypes
import smtplib
import socket
import requests
from struct import unpack

try:
    from email import MIMEText, MIMEMultipart
except ImportError:
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText

_logger = logger.Logger("Mailer")

_IP = None


def ip_is_local(ip):
    """Determines if ip is local

    Code from http://stackoverflow.com/a/8339939/1099682

    :param ip: and ip-adress
    :type ip : str
    :return: bool
    """
    f = unpack('!I', socket.inet_pton(socket.AF_INET, ip))[0]
    private = (
        [2130706432, 4278190080],  # 127.0.0.0,   255.0.0.0   http://tools.ietf.org/html/rfc3330
        [3232235520, 4294901760],  # 192.168.0.0, 255.255.0.0 http://tools.ietf.org/html/rfc1918
        [2886729728, 4293918720],  # 172.16.0.0,  255.240.0.0 http://tools.ietf.org/html/rfc1918
        [167772160,  4278190080],  # 10.0.0.0,    255.0.0.0   http://tools.ietf.org/html/rfc1918
    )
    for net in private:
        if (f & net[1]) == net[0]:
            return True
    return False


def get_my_ip():
    global _IP
    if _IP:
        return _IP

    try:
        _IP = [(s.connect(('8.8.8.8', 80)), s.getsockname()[0], s.close())
              for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except IndexError:
        _logger.info("Failed to get IP via socket")

    if _IP is None or ip_is_local(_IP):
        try:
            _IP = requests.request('GET', 'http://myip.dnsomatic.com').text
        except requests.ConnectionError:
            _logger.info("Failed to get IP via external service provider")
            _IP = None

    return _IP


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
        try:
            server.connect()
        except socket.error:
            return None

    return server


def can_get_server_with_current_settings():

    if AppConfig().mail.server:
        server = mail.get_server(AppConfig().mail.server, smtp_port=AppConfig().mail.port,
                                 login=AppConfig().mail.user, password=AppConfig().mail.password)
    else:
        server = None

    return server is not None


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
        return False

    if not sender:
        sender = get_default_email()

    try:
        msg = MIMEMultipart()
    except TypeError:
        msg = MIMEMultipart.MIMEMultipart()

    msg['From'] = sender
    msg['To'] = receiver if isinstance(receiver, StringTypes) else ", ".join(receiver)
    msg['Subject'] = subject
    try:
        msg.attach(MIMEText(message))
    except TypeError:
        msg.attach(MIMEText.MIMEText(message))

    if isinstance(receiver, StringTypes):
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

    return True


def get_host_name():

    try:
        return socket.gethostbyaddr(get_my_ip())[0]
    except (IndexError, socket.herror):
        return None


def get_default_email(username="no-reply---scan-o-matic"):

    hostname = get_host_name()
    if not hostname:
        hostname = "scanomatic.somewhere"

    return "{0}@{1}".format(username, hostname)
