# syntax=docker/dockerfile:1
FROM python:latest
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN pip install pickle4 && pip install flask && pip install numpy
EXPOSE 3000
COPY . .
CMD ["flask", "run"]
