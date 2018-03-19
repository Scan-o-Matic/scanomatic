from __future__ import absolute_import

from flask import Response, request

from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest,
    multiprocess
)

REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Number of requests by method, status and url rule',
    ['path', 'http_status', 'method'],
)


class Prometheus(object):

    def __init__(self, app):
        self.init_app(app)

    def init_app(self, app):
        app.after_request(self._request_counting)

        @app.route("/metrics")
        def metrics():
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
            return Response(data, mimetype=CONTENT_TYPE_LATEST)

    def _request_counting(self, response):
        REQUEST_COUNTER.labels(
            request.url_rule if request.url_rule else request.url,
            response.status_code, request.method
        ).inc()
        return response
