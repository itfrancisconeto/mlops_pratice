FROM python:latest
RUN pip install pickle4 && pip install flask && pip install numpy
COPY . /app
WORKDIR /app
EXPOSE 3000
CMD python app.py