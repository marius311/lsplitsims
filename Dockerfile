FROM debian:jessie 

RUN apt-get update \
    && apt-get install -y \ 
        curl \
        cython \
        gfortran-4.9 \
        python-matplotlib \
        python-numpy \
        python-pip \
        python-pyfits \
        python-scipy \
    && pip install jupyter pypico \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-4.9 20


WORKDIR /root

# install cosmoslik/mspec
RUN mkdir cosmoslik mspec \
    && curl -L https://github.com/marius311/cosmoslik/tarball/cd23e4a | tar zxf - -C cosmoslik --strip=1 \
    && curl -L https://github.com/marius311/mspec/tarball/ea533f5     | tar zxf - -C mspec     --strip=1
ENV PYTHONPATH=/root/mspec:/root/cosmoslik:$PYTHONPATH

# install camb
RUN mkdir camb \
    && curl -L https://github.com/marius311/camb/tarball/616191d      | tar zxf - -C camb      --strip=1 \
    && cd camb/pycamb \
    && python setup.py install

WORKDIR /root/shared
CMD jupyter-notebook --ip=* --no-browser
