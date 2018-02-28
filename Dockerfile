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

COPY data/ /tmp/data/
COPY scripts/ /tmp/scripts/
COPY scanomatic/ /tmp/scanomatic/
COPY setup.py /tmp/setup.py

COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/ccc.js /tmp/scanomatic/ui_server_data/js/ccc.js
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/scanning.js /tmp/scanomatic/ui_server_data/js/scanning.js
RUN cd /tmp && python setup.py install

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PGPASSFILE=/etc/scanomatic/pgpass
ENTRYPOINT ["/entrypoint.sh"]
CMD scan-o-matic --no-browser
