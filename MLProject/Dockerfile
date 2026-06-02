FROM python:3.10-slim 
WORKDIR /app 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
COPY modelling.py . 
COPY titanic_preprocessed/ ./titanic_preprocessed/ 
CMD ["python", "modelling.py"] 
