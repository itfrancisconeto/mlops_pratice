# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app

COPY ./api/requirements.txt requirements.txt
RUN pip install --no-dependencies -r requirements.txt

COPY . .

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
