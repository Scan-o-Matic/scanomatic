from __future__ import absolute_import
from flask import request
from prometheus_client import (
    Counter,
    start_http_server,
)


REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Number of requests by method, status and url rule',
    ['path',  'http_status', 'method']
)


class Prometheus(object):
    def __init__(self, app):
        self.init_app(app)

    def init_app(self, app):
        app.after_request(self._request_counting)

    def _request_counting(self, response):
        REQUEST_COUNTER.labels(
            request.url_rule if request.url_rule else request.url,
            response.status_code, request.method).inc()
        return response

    def start_server(self, port):
        start_http_server(port)
