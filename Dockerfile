FROM python:3.12-slim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY gnsm ./gnsm
RUN python -m pip install --no-cache-dir .

CMD ["python", "-m", "gnsm", "demo"]
