from __future__ import absolute_import

import json

import requests


def post_json(url, payload):

    return requests.post(
        url,
        headers={"Content-Type": 'application/json'},
        data=json.dumps(payload)
    )
