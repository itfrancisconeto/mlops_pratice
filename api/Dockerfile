# syntax=docker/dockerfile:1
FROM python:latest
WORKDIR /code
RUN pip install pickle4 && pip install flask && pip install numpy
EXPOSE 3000
COPY . .
CMD ["python", "/code/api/app.py"]