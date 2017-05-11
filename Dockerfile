FROM python:2.7

COPY requirements.txt /tmp/requirements.txt
COPY data/ /tmp/data/
COPY scripts/ /tmp/scripts/
COPY scanomatic/ /tmp/scanomatic/
COPY setup.py /tmp/setup.py
COPY setup_tools.py /tmp/setup_tools.py

RUN pip install -r /tmp/requirements.txt
RUN cd /tmp && python setup.py install --default

CMD scan-o-matic --no-browser --global

EXPOSE 5000
