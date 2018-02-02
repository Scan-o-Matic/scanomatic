from flask import Response, request
from prometheus_client import (
    multiprocess, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST,
    Counter
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
        app.add_url_rule('/metrics', view_func=self._metrics_view)
        app.after_request(self._request_counting)

    def _metrics_view(self):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    def _request_counting(response):
        REQUEST_COUNTER.labels(
            request.url_rule if request.url_rule else request.url,
            response.status_code, request.method).inc()
        return response
