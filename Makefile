default: build run

build:
	docker build -t lsplitsims .


squash:
	docker save lsplitsims | sudo docker-squash -t lsplitsims_squashed | docker load

push:
	docker tag -f lsplitsims marius311/lsplitsims
	docker push marius311/lsplitsims

nb:
	docker run -dtp 8888:8888 --name lsplitsims -v `pwd`/shared:/root/shared -v `pwd`/../../paper/plots:/root/shared/plots lsplitsims sh -c "jupyter-notebook --ip=* --no-browser"

run:
	docker run --rm -it-v `pwd`/run_sim.py:/root/run_sim.py -v `pwd`/shared:/root/shared lsplitsims $(CMD)

clean:
	docker rm -f lsplitsims
