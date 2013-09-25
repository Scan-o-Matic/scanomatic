#!/usr/bin/env python
"""Socket-based file transfer"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import socket
import SocketServer
from time import sleep

#
# EXCEPTIONS
#

class NotACorrectPath(Exception): pass

#
# GLOBALS
#

GREET_TEXT = "NwJEmle0itBT"
RESPONSE_TEXT = "1i3P1x8E1OnV"
PORT = 5500
BUFF_SIZE = 4096
COMPLETE_TEXT = ["You are refused", "File transferred", "Nothing recieved"]

#
# FUNCTIONS
#

def get_my_ip(remote_addr='google.com'):

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((remote_addr, 0))
    return s.getsockname()[0]


def _connect(IP, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.settimeout(2)
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
        sleep(0.05)

def _recieve_msg(sock):
    msg = ""
    while True:
        tmp_msg = sock.recv(BUFF_SIZE)
        msg += tmp_msg
        if len(tmp_msg) < BUFF_SIZE:
            break

    return msg

def _greet(sock):

   if _recieve_msg(sock) == GREET_TEXT:
       _send_msg(sock, RESPONSE_TEXT)
       if _recieve_msg(sock) == "1":
           return True
       else:
           return False
   else:
       return False

def send_file(IP, f_path, port=PORT):

    send_success = False
    try:
        fh = open(f_path, 'r')
    except:
        raise NotACorrectPath(f_path)

    sock = _connect(IP, port)

    if _greet(sock):
        
        prev = -1
        cur = 0
        while prev < cur:
            prev = cur
            msg = fh.read(BUFF_SIZE)
            _send_msg(sock, msg)
            cur = fh.tell()
            print "Sending chunk size", len(msg), "Total sent {0}kb".format(cur / 1024)
        send_success = True

        print "All was sent"
        #_send_msg(sock, "")

    fh.close()
    transfer_success = _recieve_msg(sock)
    sock.close()
    return transfer_success

class EchoRequestHandler(SocketServer.BaseRequestHandler):

    def setup(self):

        print self.client_address, 'connected!'
        self.request.send(GREET_TEXT, BUFF_SIZE)
        msg = self.request.recv(BUFF_SIZE)
        if msg == RESPONSE_TEXT:
            self._connection = True
            self._accepted = 1
            print "Connection accepted"
        else:
            self._connection = False
            self._accepted = 0
            print "Connection denied"

        self.request.send(str(self._accepted), BUFF_SIZE)

    def handle(self):

        #Recieve file
        data = ""
        while self._connection:
            msg = self.request.recv(BUFF_SIZE)
            data += msg
            print "Got chunk of size {0} total size is {1}kb".format(len(msg), len(data)/1024)
            if len(msg) < BUFF_SIZE:
                self._connection = False

        #Save file
        if data != "":
            print "Will now save file of size {0}kb".format(len(data) / 1024)
            try:
                fh = open('test.data', 'w')
                fh.write(data)
                fh.close()
            except:
                self._accepted = 2
                print "Could not save"

        elif data=="" and self._accepted==1:
            self._accepted = 2

    def finish(self):
        print self.client_address, 'disconnected!\n'
        self.request.send(COMPLETE_TEXT[self._accepted],
                BUFF_SIZE)


#server host is a tuple ('host', port)
if __name__ == "__main__":

    IP = get_my_ip()
    server = SocketServer.ThreadingTCPServer((IP, PORT), EchoRequestHandler)
    print "\n\nServer is up an listening to {0}:{1}\n".format(IP, PORT)
    server.serve_forever()
