#!/usr/bin/env python

import os
import sh
import src.resource_path as resource_path

desktop = """[Desktop Entry]
Name=Scan-o-Matic Session
Comment=Custom launch-script for scannomatic computers
Exec={0}
X-Ubuntu-Gettext-Domain=gdm"""

xsession = """#! /bin/bash
{0} &
gnome-screensaver-command --lock &
gnome-session --session=gnome
"""

scanomatic_proc = """#!/bin/bash
# chkconfig: 0356 99 1 
# description: Scan-o-Matic daemon
# processname: scanomatic
#   /etc/rc.d/init.d/scanomatic
 
# Source function library.
. /etc/init.d/functions
 
#<define any local shell functions used by the code that follows>

name='Scan-o-Matic daemon'

# Functions
 
start() {
    echo -n "Starting $name"
    #start daemons, perhaps with the daemon function>

    success
}   
 
stop() {
    echo -n "Stopping $name"

    success
}


# Run-behaviour
 
case "$1" in
    start)
    start
    ;;
    stop)
    stop
    ;;
    status)
    echo "Not applied to service"
    ;;
    restart)
    echo "Not applied to service"
    ;;
    reload) 
    echo "Not applied to service"
    ;;
    condrestart)
    echo "Not applied to service"
    ;;
    probe)
    ;;
    *)
    echo "Usage: scanomatic{start|stop|status|reload|restart[|probe]"
    exit 1
    ;;
esac
exit $?"""

paths = resource_path.Paths()

xsession_file_path = os.path.join(
    os.path.expanduser('~'), '.xsession_scanomatic')


#write the xsession file
fh = open(xsession_file_path, 'w')
fh.write(xsession.format(paths.revive))
fh.close()
chmod = sh.chmod.bake("+x")
chmod(xsession_file_path)

#make temp desktopfile
desktop_path = os.path.join(paths.config, 'tmp.desktop')

fh = open(desktop_path, 'w')
fh.write(desktop.format(xsession_file_path))
fh.close()
os.system(
    'gksu mv {0} /usr/share/xsessions/scanomatic.desktop'.format(
    desktop_path))
