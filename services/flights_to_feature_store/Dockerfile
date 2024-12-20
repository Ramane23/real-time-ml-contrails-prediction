FROM python:3.10.3-slim-buster

# Install gcc, python3-dev, and other necessary packages
RUN apt-get update && apt-get install -y \
    gcc wget build-essential \
    python3-dev libssl-dev libffi-dev \
    ca-certificates python3-pip

# Stream output to the console
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN pip install poetry==1.8.2

# Set the working directory inside the container
WORKDIR /app

# add /app/src to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock /app/

# copy the tools directory into the container so that we can install it
#COPY /tools /app/tools

# Install Python dependencies from the pyproject.toml file
RUN poetry install

# Copy all the source code into the container
COPY . /app

# Command to run your application
CMD ["poetry", "run", "python", "src/main.py"]
