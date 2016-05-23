This is the code for the `planck_param_sims` application at [Cosmology@Home](http://www.cosmologyathome.org). To build the Docker container:

    make build

You can then run it with,

    docker run lsplitsims

To reduce file size (likely only needed if you're actually going to run it at Cosmology@Home), you need [docker-stfd](https://github.com/marius311/stfd) installed and run:

    make squash
