# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Etc/UTC
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN apt update && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York apt install -y libopencv-dev python3-opencv libxcb-xinerama0

WORKDIR /app
COPY . /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "Bug MLOps Project/src/main.py"]
