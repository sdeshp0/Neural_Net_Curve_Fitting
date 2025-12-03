include .env

.PHONY: build up down logs clean

build:
	docker compose -f docker-compose.yaml build

up:
	docker compose -f docker-compose.yaml --env-file .env up -d

down:
	docker compose -f docker-compose.yaml down

logs:
	docker compose -f docker-compose.yaml logs -f

clean:
	docker compose -f docker-compose.yaml down -v --remove-orphans
	docker image prune -f
