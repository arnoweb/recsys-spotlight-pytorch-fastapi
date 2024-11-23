FROM python:3.11-slim-bookworm

RUN apt update && apt install git -y
RUN pip install solara --no-cache-dir

WORKDIR /srv
# Caching Introduced here
COPY requirements.txt /srv/
RUN pip install -r requirements.txt --no-cache-dir


COPY . /srv

CMD ["solara", "run", "app.py", "--port=80", "--host=0.0.0.0", "--production" ]