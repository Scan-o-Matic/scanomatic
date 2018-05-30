FROM node:4 as npmbuilder
RUN mkdir /src
WORKDIR /src

COPY package.json /src
RUN npm install

COPY scanomatic/ui_server_data /src/scanomatic/ui_server_data
COPY webpack.config.js /src
COPY .babelrc /src
RUN npm run build



FROM ubuntu:16.04
RUN apt update && apt -y install python-pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install gunicorn

COPY data/ /tmp/data/
COPY scripts/ /tmp/scripts/
COPY scanomatic/ /tmp/scanomatic/
COPY setup.py /tmp/setup.py

RUN mkdir /var/run/prometheus_multiproc
ENV prometheus_multiproc_dir=/var/run/prometheus_multiproc

COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/ccc.js /tmp/scanomatic/ui_server_data/js/ccc.js
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/scanning.js /tmp/scanomatic/ui_server_data/js/scanning.js
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/projects.js /tmp/scanomatic/ui_server_data/js/projects.js
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/statuspage.js /tmp/scanomatic/ui_server_data/js/statuspage.js
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/qc.js /tmp/scanomatic/ui_server_data/js/qc.js
RUN cd /tmp && python setup.py install

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PGPASSFILE=/etc/scanomatic/pgpass
ENV WEB_CONCURRENCY=2
ENTRYPOINT ["/entrypoint.sh"]
CMD gunicorn \
    --config python:scanomatic.ui_server.gunicorn_config \
    --bind 0.0.0.0:5000 \
    "scanomatic.ui_server.ui_server:create_app()"
