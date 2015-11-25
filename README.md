To build Dockerfile and run:

    docker build -t run_sim .
    docker run --rm -it run_sim [--real]

To reduce file size (only needed e.g. for Cosmology@Home):

    docker save run_sim | sudo docker-squash -from bf84c1d84a8f -t run_sim_reallysmall | docker load

See also https://github.com/jwilder/docker-squash
