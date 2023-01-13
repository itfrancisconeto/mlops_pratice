FROM python:latest
RUN pip install pickle4 && pip install flask && pip install numpy
WORKDIR /app
COPY . /app/
EXPOSE 3000
CMD ["python", "/app/app.py"]