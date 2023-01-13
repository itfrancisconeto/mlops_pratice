FROM python:latest
WORKDIR /app
RUN pip install pickle4 && pip install flask && pip install numpy
COPY . /app
EXPOSE 3000
CMD python app.py