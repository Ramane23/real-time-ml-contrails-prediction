FROM python:3.10.3-slim-buster

# install gcc and python3-dev to compile Python packages
RUN apt-get update && apt-get install -y \
    gcc wget build-essential \
    python3-dev libssl-dev libffi-dev \
    ca-certificates

# stream output to console
ENV PYTHONUNBUFFERED=1

# install poetry inside the container
RUN pip install poetry==1.8.2

WORKDIR /app

# This line here is ugly as hell. Please don't do this.
# Overwriting the PYTHONPATH environment variable is a bad practice.
# add /app/src to PYTHONPATH
# ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock /app/

# copy the pip.conf file into the container
COPY pip.conf /etc/pip.conf

# install Python dependencies from the pyproject.toml file
RUN poetry install 

# copy all the source code into the container and the README file
COPY . /app

# Expose the port that the application will run on
EXPOSE 8000

# Command to run the application with Uvicorn
CMD ["poetry", "run", "uvicorn", "src.restapi:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]