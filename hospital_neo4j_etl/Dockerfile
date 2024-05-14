FROM python:3.11-slim

WORKDIR /app

# copy the contents from hospital_neo4j_etl/src/ into the /app directory within the container
COPY ./src/ /app

# install the dependencies from pyproject.toml
COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

# run entrypoint.sh
CMD ["sh", "entrypoint.sh"]