FROM python:3.7
LABEL maintainer="n_nedelkina@mail.ru"
COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONPATH=.
