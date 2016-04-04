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


# and see https://github.com/docker/docker/issues/14651
RUN rm -rf /root && mkdir -p /root/shared
WORKDIR /root


#install healpy (w/o matplotlib dep)
RUN curl -L https://pypi.python.org/packages/source/h/healpy/healpy-1.9.1.tar.gz | tar zxf - \
    && cd healpy-1.9.1 \
    && python ez_setup.py \
    && sed -ie '55,57d' healpy/__init__.py \
    && python setup.py develop -N \
    && rm -rf build cfitsio healpixsubmodule
ENV PYTHONPATH=/root/healpy-1.9.1

#install quickbeam (for kernel computation)
RUN mkdir quickbeam \
    && curl -L https://github.com/marius311/quickbeam/tarball/0f61a97 | tar zxf - -C quickbeam --strip-components=1 \
    && cd quickbeam \
    && python setup.py install

# install cosmoslik/mspec
RUN mkdir cosmoslik mspec \
    && curl -L https://github.com/marius311/cosmoslik/tarball/cd23e4a | tar zxf - -C cosmoslik --strip-components=1 \
    && curl -L https://github.com/marius311/mspec/tarball/ea533f5     | tar zxf - -C mspec     --strip-components=1
ENV PYTHONPATH=/root/mspec:/root/cosmoslik:$PYTHONPATH

# install camb
RUN mkdir camb \
    && curl -L https://github.com/cmbant/camb/tarball/a28e487         | tar zxf - -C camb      --strip-components=1 \
    && sed -i '/march=native/d' camb/Makefile \
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
    planck_2_2500.covmat \
    commander_dx11d2_mask_temp_n0016_likelihood_v1.fits \
    commander_dx11d2_mask_temp_n0016_likelihood_v1_f.dat \
    run_sim.py \
    /root/
COPY covs /root/covs
RUN mkdir -p /root/shared/results
ENV PYTHONPATH=/root:$PYTHONPATH


WORKDIR /root

ENTRYPOINT ["python","run_sim.py"]
