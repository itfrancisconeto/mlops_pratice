FROM ubuntu

COPY ./api/requirements.txt requirements.txt
RUN pip install --no-dependencies -r requirements.txt

WORKDIR /app

COPY . .

CMD [ "python","-m","flask","run","--host=0.0.0.0" ]