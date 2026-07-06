FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python

WORKDIR /opt/cairn
COPY requirements.txt ./
RUN python -m pip install --upgrade "pip==23.0" \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["python", "-m"]
CMD ["cairn.routes.train", "--route", "beta/standard.yaml"]
