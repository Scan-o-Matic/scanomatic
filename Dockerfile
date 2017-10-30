FROM node:4 as npmbuilder
COPY . /src
WORKDIR /src
RUN npm install
RUN npm run build

FROM ubuntu:16.04
RUN apt update && apt -y install usbutils software-properties-common python-pip
# net-tools & iputils-ping are used in the xml-writer which should be removed soon
RUN apt -y install net-tools iputils-ping
RUN echo -e "\n" | add-apt-repository ppa:rolfbensch/sane-release
RUN apt-get update
RUN apt -y install libsane=1.0.27-xenial1 sane-utils=1.0.27-xenial1 libsane-common=1.0.27-xenial1
# Add scanner id to sane config in case scanimage -L cannot find the scanner automatically
# Epson V800
RUN echo "usb 0x4b8 0x12c" >> /etc/sane.d/epson2.conf
# Epson V700
RUN echo "usb 0x4b8 0x151" >> /etc/sane.d/epson2.conf

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY data/ /tmp/data/
COPY scripts/ /tmp/scripts/
COPY scanomatic/ /tmp/scanomatic/
COPY setup.py /tmp/setup.py
COPY setup_tools.py /tmp/setup_tools.py
COPY get_installed_version.py /tmp/get_installed_version.py
COPY --from=npmbuilder /src/scanomatic/ui_server_data/js/ccc.js /tmp/scanomatic/ui_server_data/js/ccc.js

RUN cd /tmp && python setup.py install --default
CMD scan-o-matic --no-browser
EXPOSE 5000
