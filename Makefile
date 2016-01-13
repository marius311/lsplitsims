default: build run

build:
	docker build -t lsplitsims .

nb:
	docker run -dtp 8888:8888 --name lsplitsims -v `pwd`/shared:/root/shared lsplitsims

run:
	docker run --rm -it-v `pwd`/shared:/root/shared lsplitsims $(CMD)

clean:
	docker rm -f lsplitsims
