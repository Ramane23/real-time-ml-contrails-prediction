run-dev:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run python src/main.py
	
run-dev-historical:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/main.py

build:
	docker build -t flights-with-weather .

run: build
	docker run \
		--name flights_with_weather_instance \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_INPUT_TOPIC_NAME=live_flights \
		-e KAFKA_CONSUMER_GROUP=live_flights_consumer \
		-e KAFKA_OUTPUT_TOPIC_NAME=live_flights_with_weather \
		-e AVIATION_EDGE_API_KEY=${AVIATION_EDGE_API_KEY} \
		-e METEOMATICS_USERNAME=${METEOMATICS_USERNAME} \
		-e METEOMATICS_PASSWORD=${METEOMATICS_PASSWORD} \
		flights-with-weather

run-historical: build
	docker volume create flights-with-weather-volume
	docker run \
		--name historical_flight_with_weather_instance \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_INPUT_TOPIC_NAME=historical_flights \
		-e KAFKA_CONSUMER_GROUP=historical_flights_consumer \
		-e KAFKA_OUTPUT_TOPIC_NAME=historical_flights_with_weather \
		-e AVIATION_EDGE_API_KEY=${AVIATION_EDGE_API_KEY} \
		-e METEOMATICS_USERNAME=${METEOMATICS_USERNAME} \
		-e METEOMATICS_PASSWORD=${METEOMATICS_PASSWORD} \
		flights-with-weather

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format