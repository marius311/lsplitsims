FROM debian:jessie 

RUN apt-get update \
    && apt-get install -y \ 
        curl \
        python-matplotlib \
        python-numpy \
        python-pip \
        python-pyfits \
        python-scipy \
    && pip install jupyter pypico

WORKDIR /root

# install cosmoslik/mspec
RUN mkdir cosmoslik mspec \
    && curl -L https://github.com/marius311/cosmoslik/tarball/cd23e4a | tar zxf - -C cosmoslik --strip=1 \
    && curl -L https://github.com/marius311/mspec/tarball/ea533f5     | tar zxf - -C mspec     --strip=1
ENV PYTHONPATH=/root/mspec:/root/cosmoslik:$PYTHONPATH

WORKDIR /root/shared
CMD jupyter-notebook --ip=* --no-browser
