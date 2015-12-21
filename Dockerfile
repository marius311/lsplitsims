FROM debian:jessie 

RUN apt-get update \
    && apt-get install -y \
        curl \
        cython \
        gfortran-4.9 \
        git \
        libcfitsio-dev \
        liblapack-dev \
        make \
        python \
        python-numpy \
        python-pyfits \
        python-scipy \
        python-six \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-4.9 20

# needed for interactive work:
RUN apt-get update \
    && apt-get install -y \ 
        python-pip \
        python-matplotlib \
    && pip install jupyter pypico
EXPOSE 8888

# so the root-created files below are read/write-able by default
# and see https://github.com/docker/docker/issues/14651
RUN umask a+rw && rm -rf /root && mkdir /root-created
WORKDIR /root


#install healpy
# RUN curl -L https://pypi.python.org/packages/source/h/healpy/healpy-1.9.1.tar.gz | tar zxf - \
#     && cd healpy-1.9.1 \
#     && python ez_setup.py \
#     && python setup.py build


# install cosmoslik/mspec
RUN mkdir cosmoslik mspec camb \
    && curl -L https://github.com/marius311/cosmoslik/archive/cd23e4a.tar.gz | tar zxf - -C cosmoslik --strip-components=1 \
    && curl -L https://github.com/marius311/mspec/archive/112ca2f.tar.gz     | tar zxf - -C mspec     --strip-components=1
ENV PYTHONPATH=/root/mspec:/root/cosmoslik

# install camb
RUN    curl -L https://github.com/cmbant/camb/archive/a28e487.tar.gz         | tar zxf - -C camb      --strip-components=1 \
    && cd camb/pycamb \
    && python setup.py install

# install clik
ADD COM_Likelihood_Code-v2.0.R2.00.tar.bz2 /root
RUN cd /root/plc-2.0 \
    && ./waf configure \
    && ./waf install
ENV PYTHONPATH=/root/plc-2.0/lib/python2.7/site-packages:$PYTHONPATH

# copy data files and script
ADD commander_rc2_v1.1_l2_29_B.clik.tgz \
    plik_lite_v18_TT.clik.tgz \
    base_plikHM_TT_tau07.minimum.theory_cl \
    run_sim.py \
    /root/
ENV PYTHONPATH=/root:$PYTHONPATH


# cleanup build packages:
# 
# RUN apt-get remove -y --purge \
#         curl \
#         cython \
#         gcc \
#         gcc-4.9 \
#         gfortran-4.9 \
#         git \
#         libcfitsio-dev \
#         liblapack-dev \
#         make \
#     && apt-get autoremove --purge -y \
#     && apt-get update \
#     && apt-get install libgomp1 \
#     && rm -rf /var/lib/apt/lists/* /root/plc-2.0/build /root/plc-2.0/src /usr/share


CMD jupyter-notebook --ip=* --no-browser
