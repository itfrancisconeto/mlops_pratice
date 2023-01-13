# syntax=docker/dockerfile:1
FROM python:latest
WORKDIR /code/api
RUN pip install pickle4 && pip install flask && pip install numpy
EXPOSE 3000
COPY . .
CMD ["python", "app.py"]