all: build run bash

build:
	docker build -f Dockerfile-dev --tag gpt-neox:cuda121 --progress plain .

build-cuda117:
	docker build -f Dockerfile-cuda117 --tag gpt-neox:cuda117 .

run:
	docker compose -f docker-compose-dev.yml up -d

run-cuda117:
	docker compose -f docker-compose-dev-cuda117.yml up -d

bash:
	docker exec -it gpt-neox bash

bash-cuda117:
	docker exec -it gpt-neox-cuda117 bash

############
# Teardown #
############

stop:
	docker compose -f docker-compose-dev.yml stop
	docker compose -f docker-compose.yml stop

clean:
	docker compose -f docker-compose-dev.yml down
	docker compose -f docker-compose.yml down

down: clean