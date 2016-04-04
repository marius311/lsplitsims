build:
	docker build -t marius311/plancklcdm .

start:
	docker run -dtp 8888 --name plancklcdm -v `pwd`:/root/shared -v `pwd`/../paper/plots:/plots marius311/plancklcdm
	@echo "Now point browser to $$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' plancklcdm):8888"

stop:
	docker rm -f plancklcdm
