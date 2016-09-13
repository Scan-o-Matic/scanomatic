import os
from multiprocessing import Process
from time import sleep
from subprocess import call, STDOUT

from scanomatic.io.logger import Logger

_logger = Logger("Daemonizer")


def _daemon_process(path_to_exec, std_out_path, args, shell):

    with open(std_out_path, 'w') as fh:
        args = list(str(a) for a in args)

        if shell:
            fh.write("*** LAUNCHING IN SHELL: {0} ***\n\n".format(" ".join([path_to_exec] + list(args))))
            retcode = call(" ".join([path_to_exec] + args), stderr=STDOUT, stdout=fh, shell=True)
        else:
            fh.write("*** LAUNCHING WITHOUT SHELL: {0} ***\n\n".format([path_to_exec] + list(args)))
            retcode = call([path_to_exec] + args, stderr=STDOUT, stdout=fh, shell=False)

        if retcode:
            fh.write("\n*** DAEMON EXITED WITH CODE {0} ***\n".format(retcode))
        else:
            fh.write("\n*** DAEMON DONE ***\n")


def daemon(path_to_executable, std_out=os.devnull, daemon_args=tuple(), shell=True):

    _logger.info("Launching daemon {0} (args={2}, {3}), outputting to {1} ".format(
        path_to_executable, std_out, daemon_args, "shell" if shell else "no shell"
    ))
    d = Process(name='daemon', target=_daemon_process, args=(path_to_executable, std_out, daemon_args, shell))
    d.daemon = True
    d.start()

    sleep(1)