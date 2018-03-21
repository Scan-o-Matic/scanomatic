from .exceptions import UnknownScanjobError


class DeleteScanjobError(Exception):
    pass


def delete_scanjob(scanjob_store, scanjob_id):
    try:
        scanjob = scanjob_store.get_scanjob_by_id(scanjob_id)
    except LookupError:
        raise UnknownScanjobError('No scanjob with id "{}"'.format(scanjob_id))
    if scanjob.start_time is not None:
        raise DeleteScanjobError(
            'Scanjob {} has been started, cannot delete'.format(scanjob_id)
        )
    scanjob_store.delete_scanjob(scanjob_id)
