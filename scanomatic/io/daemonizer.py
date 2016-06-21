import os
from multiprocessing import Process
from subprocess import Popen, STDOUT
from time import sleep


def daemon_process(path_to_exec, std_out_path, args, shell):

    with open(std_out_path, 'w') as fh:
        args = (str(a) for a in args)

        if shell:
            fh.write("*** LAUNCHING IN SHELL: {0} ***\n\n".format(" ".join([path_to_exec] + list(args))))
            p = Popen(" ".join([path_to_exec] + list(args)), stderr=STDOUT, stdout=fh, shell=True)
        else:
            fh.write("*** LAUNCHING WITHOUT SHELL: {0} ***\n\n".format([path_to_exec] + list(args)))
            p = Popen([path_to_exec] + list(args), stderr=STDOUT, stdout=fh, shell=False)

        p.wait()


def daemon(path_to_executable, std_out=os.devnull, daemon_args=tuple(), shell=True):

    _logger.info("Launching daemon {0} (args={2}, {3}), outputting to {1} ".format(
        path_to_executable, std_out, daemon_args, "shell" if shell else "no shell"
    ))
    d = Process(name='daemon', target=daemon_process, args=(path_to_executable, std_out, daemon_args, shell))
    d.daemon = True
    d.start()
    sleep(5)