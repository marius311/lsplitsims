build:
	docker build -t plancklcdm .

run:
	docker run --rm -tp 8888:8888 -v `pwd`:/root/shared -v `pwd`/../paper/plots:/plots plancklcdm
