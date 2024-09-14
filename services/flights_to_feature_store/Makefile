run-dev:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run python src/main.py
	
run-dev-historical:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/main.py

build:
	docker build -t flights-to-feature-store .

run: build
	docker run \
		--name flights_with_weather_instance \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_TOPIC_NAME=live_flights_with_weather \
		-e KAFKA_CONSUMER_GROUP=live_flights_with_weather_consumer \
		-e LIVE_OR_HISTORICAL=live \
		-e HOPSWORKS_API_KEY=${HOPSWORKS_API_KEY} \
		-e HOPSWORKS_PROJECT_NAME=${HOPSWORKS_PROJECT_NAME} \
		-e FEATURE_GROUP_NAME=${FEATURE_GROUP_NAME} \
		-e FEATURE_GROUP_VERSION=${FEATURE_GROUP_VERSION} \
		-e FEATURE_VIEW_NAME=${FEATURE_VIEW_NAME} \
		-e FEATURE_VIEW_VERSION=${FEATURE_VIEW_VERSION} \
		flights-to-feature-store

run-historical: build
	docker volume create flights-to-feature-store-volume
	docker run \
		--name flights_with_weather_instance \
		--network=redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
		-e KAFKA_TOPIC_NAME=historical_flights_with_weather \
		-e KAFKA_CONSUMER_GROUP=historical_flights_with_weather_consumer \
		-e LIVE_OR_HISTORICAL=historical \
		-e HOPSWORKS_API_KEY=${HOPSWORKS_API_KEY} \
		-e HOPSWORKS_PROJECT_NAME=${HOPSWORKS_PROJECT_NAME} \
		-e FEATURE_GROUP_NAME=${FEATURE_GROUP_NAME} \
		-e FEATURE_GROUP_VERSION=${FEATURE_GROUP_VERSION} \
		-e FEATURE_VIEW_NAME=${FEATURE_VIEW_NAME} \
		-e FEATURE_VIEW_VERSION=${FEATURE_VIEW_VERSION} \
		flights-to-feature-store
lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .

lint-and-format: lint format