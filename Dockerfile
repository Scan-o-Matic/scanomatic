FROM node:4 as npmbuilder
COPY scanomatic/ui_server_data /src/scanomatic/ui_server_data
COPY package.json /src
COPY webpack.config.js /src
COPY .babelrc /src
WORKDIR /src
RUN npm install
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

RUN cd /tmp && python setup.py install --default
CMD scan-o-matic --no-browser

