# FROM python:3.7-slim-buster
FROM ubuntu:18.04

WORKDIR /app

# install ffmpeg 
RUN apt-get update
RUN apt-get install -y ffmpeg

# install python
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# install python libs
RUN pip3 install tqdm==4.44.1

RUN pip3 install Pillow==8.0.1
RUN pip3 install opencv-contrib-python-headless==4.4.0.46
RUN pip3 install av==8.0.2
RUN pip3 install numpy==1.18.1
RUN pip3 install torchvision==0.8.1
RUN pip3 install torch==1.7.0

RUN pip3 install Hypercorn==0.5.4
RUN pip3 install Quart==0.6.15
RUN pip3 install Quart_CORS==0.1.3
RUN pip3 install PyYAML==5.3.1
RUN pip3 install pymongo==3.8.0
RUN pip3 install Flask_PyMongo==2.3.0
# RUN pip3 install MongoDBProxy-official==0.1.0

# install app
COPY . .
EXPOSE 80
CMD ["hypercorn", "-b", "0.0.0.0:80", "app:app"]

