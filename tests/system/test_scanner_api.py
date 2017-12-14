from StringIO import StringIO
from collections import namedtuple
import fnmatch
from hashlib import sha256
import httplib

import pytest
import requests


@pytest.fixture
def image():
    data = b'I am an image'
    return namedtuple('image', ['image', 'filename', 'digest'])(
        StringIO(data), 'my_image.tiff', sha256(data).hexdigest()
    )


def get_number_of_images(scanomatic, project):
    response = requests.get(
        scanomatic + '/api/tools/path/root/{}/'.format(project),
        params={'suffix': '.tiff', 'isDirectory': 0, 'checkHasAnalysis': 0},
    )
    suggestions = fnmatch.filter(response.json()['suggestions'], '*.tiff')
    return len(suggestions)


def test_upload_image(scanomatic, image):
    project = 'my/project'
    assert get_number_of_images(scanomatic, project) == 0
    response = requests.post(
        scanomatic + '/api/scans',
        data={
            'project': project,
            'scan_index': 5,
            'timedelta': 789,
            'digest': image.digest,
        },
        files={
            'image': (image.filename, image.image, 'image/tiff'),
        },
    )
    assert response.status_code == httplib.CREATED, response.content
    assert get_number_of_images(scanomatic, project) == 1


def test_upload_image_unknown_project(scanomatic, image):
    project = 'unknown/project'
    response = requests.post(
        scanomatic + '/api/scans',
        data={
            'project': project,
            'scan_index': 5,
            'timedelta': 789,
            'digest': image.digest,
        },
        files={
            'image': (image.filename, image.image, 'image/tiff'),
        },
    )
    assert response.status_code == httplib.BAD_REQUEST, response.content
    assert get_number_of_images(scanomatic, project) == 0
