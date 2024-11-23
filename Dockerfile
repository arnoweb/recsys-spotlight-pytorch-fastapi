FROM python:3.11-slim-bookworm

RUN apt update && apt install git -y
RUN pip install solara --no-cache-dir
# install the cpu-only torch (or any other torch-related packages)
# you might modify it to install another version
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# any packages that depend on pytorch must be installed after the previous RUN command

WORKDIR /srv
# Caching Introduced here
COPY requirements.txt /srv/
RUN pip install -r requirements.txt --no-cache-dir

RUN dvc pull model/movies_users_rating_model.pth

COPY . /srv

CMD ["solara", "run", "app.py", "--port=80", "--host=0.0.0.0", "--production" ]