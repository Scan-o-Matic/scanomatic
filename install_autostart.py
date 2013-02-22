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
