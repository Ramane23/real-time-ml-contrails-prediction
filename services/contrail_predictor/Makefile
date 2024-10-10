# Commands for training the model
train:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/training.py

# Commands for running the prediction script
predict:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/prediction.py

# Commands for running the FastAPI app with Uvicorn
restapi:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run uvicorn restapi:app --host 0.0.0.0 --port 8000 --reload

# Health check for the REST API
check-health:
	curl -X GET http://localhost:8000/healthcheck

check-health-docker:
	curl -X GET http://localhost:5005/healthcheck

# Send a valid prediction request to the local API
request:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"route": "Athens - Thessaloniki"}'

request-docker:
	curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"route": "Frankfurt - Berlin"}'

# Send a request to the production API (adjust the URL as needed)
request-production:
	curl -X POST https://contrail-predictor-contactsouley-realtimemlcontrailspr-4eda2dd1.deployments.quix.io/predict -H "Content-Type: application/json" -d '{"route": "Frankfurt - Berlin"}'

# Send an invalid prediction request to test error handling
invalid-request:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"route": "Niamey - Ouagadougou"}'

invalid-request-docker:
	curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"route": "Niamey - Ouagadougou"}'

# Build the Docker image
build:
	docker build -t contrail_predictor .

# Stop and remove existing container (if exists) before running a new one
clean:
	docker stop contrail_predictor || true && docker rm contrail_predictor || true

# Run the container after building
run: build
	docker run \
		--name contrail_predictor \
		--env-file ./setup_live_config.env \
		-p 5005:8000 \
		--name contrail_predictor \
		contrail_predictor

# Lint code with ruff
lint:
	poetry run ruff check --fix

# Format code with ruff
format:
	poetry run ruff format .

# Run both linting and formatting
lint-and-format: lint format
