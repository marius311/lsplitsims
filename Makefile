default: 

build:
	docker build -t lsplitsims .

squash:
	docker-stfd --entrypoint sh lsplitsims -c 'touch fid* covs/*/* && python run_sim.py --seeds [0] --lslices [\(2,2509\)] --dryrun && rm -rf shared'