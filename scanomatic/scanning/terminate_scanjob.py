from datetime import datetime

from pytz import utc


class TerminateScanJobError(Exception):
    pass


class UnknownScanjobError(Exception):
    pass


def terminate_scanjob(scanjob_store, scanjob_id, termination_message):
    try:
        scanjob = scanjob_store.get_scanjob_by_id(scanjob_id)
    except LookupError:
        raise UnknownScanjobError('No scan job with id {}'.format(scanjob_id))
    now = datetime.now(utc)
    if not scanjob.is_active(now):
        raise TerminateScanJobError()
    scanjob_store.terminate_scanjob(scanjob_id, now, termination_message)
