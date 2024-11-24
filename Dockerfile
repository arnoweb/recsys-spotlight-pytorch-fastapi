FROM python:3.11-slim-bookworm

#RUN apt-get update && apt-get install -y \
#    git \
#    supervisor \
#    nginx \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install git supervisor nginx -y

#RUN apt-get install git -y

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m venv /venvs/recsys-explore
RUN python -m venv /venvs/recsys-api


#RUN /venvs/recsys-explore/bin/pip install solara --no-cache-dir
# install the cpu-only torch (or any other torch-related packages)
# you might modify it to install another version
RUN /venvs/recsys-explore/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /srv
COPY requirements.txt /srv/

RUN /venvs/recsys-explore/bin/pip install -r requirements.txt --no-cache-dir

#RUN /venvs/recsys-api/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
#RUN /venvs/recsys-api/bin/pip install -r requirements.txt --no-cache-dir

COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisord.conf

COPY . /srv

EXPOSE 80

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]

#ENTRYPOINT ["sh", "start.sh", "--port", "80"]
#CMD ["solara", "run", "app.py", "--port=80", "--host=0.0.0.0", "--production" ]