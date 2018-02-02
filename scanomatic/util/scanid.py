from __future__ import absolute_import, division
from builtins import round


def generate_scan_id(scanjob, scantime):
    scan_timedelta = (scantime - scanjob.start_time)
    scan_index = round(
        scan_timedelta.total_seconds()
        / scanjob.interval.total_seconds()
    )
    return '{jobid}_{index:04.0f}_{timedelta:.4f}'.format(
        jobid=scanjob.identifier,
        timedelta=scan_timedelta.total_seconds(),
        index=scan_index,
    )
