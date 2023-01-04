FROM python:3.11

WORKDIR /proxy

COPY ./requirements.txt /proxy/requirements.txt

RUN pip install --no-cache-dir -r /proxy/requirements.txt

COPY ./app.py /proxy/app.py
COPY ./nerual_network.py /proxy/nerual_network.py

RUN apt-get update && apt-get install -y iputils-ping

CMD [ "python3", "app.py"]