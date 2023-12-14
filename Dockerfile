FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

COPY requirements.txt .

RUN apt update && \
    apt install -y --no-install-recommends python3 python3-pip && \
    pip install -r requirements.txt && \
    apt clean && \
    rm -rf /root/.cache/pip /var/lib/apt/lists/* /requirements.txt

COPY src /src

WORKDIR /src

EXPOSE 9528

ENTRYPOINT ["uvicorn", "app:app" , "--host", "0.0.0.0", "--port", "9528"]
