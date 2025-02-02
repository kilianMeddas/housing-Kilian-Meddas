FROM python:3.11

# Create the output directory and set permissions
RUN mkdir /output && chmod -R 777 /output

# Copy files into the container
COPY api.py .
COPY requirements.txt . 
COPY housing.csv .
# Install dependencies
RUN pip3 install -r requirements.txt

# Set the default command to run the API
CMD ["python", "api.py"]
