start-redpanda:
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

# Feature pipeline commands when running with live data
build-live-feature-pipeline:
	docker compose -f live-feature-pipeline.yml build

run-live-feature-pipeline: build-live-feature-pipeline
	docker compose -f live-feature-pipeline.yml up -d

stop-live-feature-pipeline:
	docker compose -f live-feature-pipeline.yml down

# Feature pipeline commands when backfilling historical data
build-historical-feature-pipeline:
	docker compose -f historical-feature-pipeline.yml build

run-historical-feature-pipeline: build-historical-feature-pipeline
	docker compose -f historical-feature-pipeline.yml up -d

stop-historical-feature-pipeline:
	docker compose -f historical-feature-pipeline.yml down

# Commands for running the training and prediction services*
build-contrail-predictor:
	docker compose -f contrail-predictor.yml build

# Run the training and prediction services
run-contrail-predictor: build-contrail-predictor
	docker compose -f contrail-predictor.yml up -d

# Stop the training and prediction services
stop-contrail-predictor:
	docker compose -f contrail-predictor.yml down

check-health-docker:
	curl -X GET http://localhost:5005/healthcheck

request-docker:
	curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"route": "Frankfurt - Berlin"}'
