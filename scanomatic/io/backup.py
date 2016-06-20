import os
import tarfile


def get_backup_name(path_to_original, max_backups=10, pattern="{0}.{1}.tar.gz"):

    t_prev = None

    for i in range(max_backups):

        path = pattern.format(path_to_original, i)
        if not os.path.isfile(path):
            return path

        s = os.stat(path)
        t = max(s.st_atime, s.st_mtime, s.st_ctime)
        if t_prev is not None and t_prev < t:
            return path
        t_prev = t

    return pattern.format(path_to_original, 0)


def backup_file(*paths):

    if not any(os.path.isfile(path) for path in paths):
        return None

    backup_path = get_backup_name(paths[0])

    with tarfile.open(backup_path, "w:gz") as tar:
        for path in paths:
            tar.add(path)

    return backup_path
