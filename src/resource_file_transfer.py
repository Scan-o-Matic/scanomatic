"""Socket-based file transfer"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import socket

#
# EXCEPTIONS
#

class NotACorrectPath(Exception): pass

#
# FUNCTIONS
#

def _connect(IP, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, port))
    return s

def _send_msg(sock, msg):
    totalsent = 0
    msglen = len(msg)
    while totalsent < msglen:
        sent = sock.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        totalsent += sent

def _recieve_msg(sock):
    totalgot = 0
    prevgot = -1
    msg = ""
    while prevgot < totalgot:
        prevgot = totalgot
        msg += sock.recv(4096)

    return msg

def _greet(sock, myKey, expectedResponse):

   _send_msg(sock, myKey)
   if _recieve_msg(sock) == expectedResponse:
       return True
   else:
       return False

def send_file(IP, port, f_path):

    send_success = False
    try:
        fh = open(f_path, 'r')
    except:
        raise NotACorrectPath(f_path)

    sock = _connect(IP, port)

    if _greet(sock, 'a', 'b'):
        
        BUFF_SIZE = 4096
        prev = -1
        cur = 0
        while prev < cur:
            prev = cur
            msg = fh.read(BUFF_SIZE)
            _send_msg(sock, msg)
            cur = fh.tell()
        send_success = True
    sock.close()
    fh.close()
