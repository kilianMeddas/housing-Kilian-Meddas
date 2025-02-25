FROM python:3.11


COPY api.py api_st.py requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "api.py"]
