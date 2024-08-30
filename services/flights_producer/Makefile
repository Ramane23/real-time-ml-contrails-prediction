run-dev:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run python src/main.py
	
run-dev-historical:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/main.py

build:
	docker build -t flight-producer .

run: build
	docker run \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_TOPIC_NAME=live_flights \
		-e LIVE_OR_HISTORICAL=live \
		-e AVIATION_EDGE_API_KEY=${AVIATION_EDGE_API_KEY} \
		flight-producer

run-historical: build
	docker volume create flight-producer-volume
	docker run \
		--name flight-producer_instance \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_TOPIC_NAME=historical_flights \
		-e LIVE_OR_HISTORICAL=historical \
		-e AVIATION_EDGE_API_KEY=${AVIATION_EDGE_API_KEY} \
		-e DAYS=1 \
		-v trade-producer-volume:/tmp/historical_trade_data \
		flight-producer

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format