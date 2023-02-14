FROM python:3.10

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY microservice /microservice

EXPOSE 8080

CMD ["uvicorn", "microservice.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]