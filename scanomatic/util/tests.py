import requests
import json


def post_json(url, payload):

    return requests.post(
        url,
        headers={"Content-Type": 'application/json'},
        data=json.dumps(payload)
    )
