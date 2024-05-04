FROM python:3.11-alpine

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "flask", "run"]