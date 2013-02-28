#!/usr/bin/env python
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
import SocketServer
from time import sleep

#
# EXCEPTIONS
#

class NotACorrectPath(Exception): pass

#
# GLOBALS
#

GREET_TEXT = "test"
RESPONSE_TEXT = "fuckyou"
PORT = 5500
COMPLETE_TEXT = ["You are refused", "File transferred", "Nothing recieved"]

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
    sleep(1)

def _recieve_msg(sock):
    totalgot = 0
    prevgot = -1
    msg = ""
    while prevgot < totalgot:
        prevgot = totalgot
        msg += sock.recv(4096)

    sleep(1)
    return msg

def _greet(sock):

   if _recieve_msg(sock) == GREET_TEXT:
       _send_msg(sock, RESPONSE_TEXT)
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

    if _greet(sock):
        
        BUFF_SIZE = 4096
        prev = -1
        cur = 0
        while prev < cur:
            prev = cur
            msg = fh.read(BUFF_SIZE)
            _send_msg(sock, msg)
            cur = fh.tell()
        send_success = True

        _send_msg(sock, "")
    fh.close()
    transfer_success = _recieve_msg(sock)
    sock.close()
    return transfer_success

class EchoRequestHandler(SocketServer.BaseRequestHandler):

    BUFF_SIZE = 4096


    def setup(self):

        print self.client_address, 'connected!'
        self.request.send(GREET_TEXT, self.BUFF_SIZE)
        msg = self.request.recv(self.BUFF_SIZE)
        if msg == RESPONSE_TEXT:
            self._connection = True
            self._accepted = 0
            print "Connection accepted"
        else:
            self._connection = False
            self._accepted = 1
            print "Connection denied"

    def handle(self):

        print "Ready to recieve file!"
        #Recieve file
        data = ""
        old_size = -1
        new_size = 0
        while self._connection:
            old_size = new_size
            data += self.request.recv(self.BUFF_SIZE)
            new_size = len(data)
            print "Chunk received", new_size, old_size
            if old_size == new_size:
                self._connection = False    

        #Save file
        if data != "":
            print "Will now save file"
            try:
                fh = open('test.data', 'w')
                fh.write(data)
                fh.close()
            except:
                self._accepted = 2
                print "Could not save"
        elif data=="" and self._accepted==1:
            sefl._accepted = 2

    def finish(self):
        print self.client_address, 'disconnected!'
        self.request.send(COMPLETE_TEXT[self._accepted],
                self.BUFF_SIZE)


#server host is a tuple ('host', port)
if __name__ == "__main__":

    server = SocketServer.ThreadingTCPServer(('localhost', PORT), EchoRequestHandler)
    server.serve_forever()
