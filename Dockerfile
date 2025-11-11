FROM python:3.11-slim-bookworm

RUN apt update && apt install -y \
    git \
    supervisor \
    nginx \
    build-essential \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    procps \
    lsof \
    net-tools \
    curl \
    systemctl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /srv
COPY requirements-docker.txt /srv/

RUN python -m venv /srv/venv-docker


#RUN /venvs/recsys-explore/bin/pip install solara --no-cache-dir
# install the cpu-only torch (or any other torch-related packages)
# you might modify it to install another version
RUN /srv/venv-docker/bin/pip install --upgrade pip setuptools wheel
RUN /srv/venv-docker/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# Spotlight's setup.py expects the legacy setuptools entrypoint, so we disable the
# default PEP 517 build isolation to keep installation compatible with pip>=25.
RUN /srv/venv-docker/bin/pip install --no-cache-dir --no-build-isolation --no-use-pep517 \
    git+https://github.com/maciejkula/spotlight.git@75f4c8c55090771b52b88ef1a00f75bb39f9f2a9


RUN /srv/venv-docker/bin/pip install --upgrade pip
RUN /srv/venv-docker/bin/pip install -r requirements-docker.txt --no-cache-dir


COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisord.conf
#COPY solara-docker/cdn /srv/venv-docker/share/solara/cdn

COPY . /srv

VOLUME ["/srv"]

RUN ln -s /srv/venv-docker/share/solara /usr/share/nginx/html/_solara

EXPOSE 80

ENV PATH="/srv/venv-docker/bin:$PATH"

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]

#ENTRYPOINT ["sh", "start.sh", "--port", "80"]
#CMD ["solara", "run", "app.py", "--port=80", "--host=0.0.0.0", "--production" ]
