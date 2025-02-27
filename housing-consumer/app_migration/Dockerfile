FROM python:3.11

COPY app_migration.py .
COPY requirements.txt . 

RUN pip3 install -r requirements.txt

CMD ["python", "app_migration.py"]
