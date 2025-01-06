run-dev:
	uv run python -m run

run-claude:
	uv run python -m llms.claude

run-ollama:
	uv run python -m llms.ollama

build:
	docker build -f Dockerfile -t news-signal .

run-with-anthropic: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e MODEL_PROVIDER=anthropic \
		--env-file anthropic_credentials.env \
		news-signal

run-with-ollama: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e MODEL_PROVIDER=ollama \
		--env-file ollama.env \
		news-signal"

setup-ollama:
	docker pull ollama/ollama
	docker network create redpanda_network || ver > NUL
	docker stop ollama || ver > NUL
	docker rm ollama || ver > NUL
	docker run -d --name ollama \
		--network redpanda_network \
		-v ollama:/root/.ollama \
		-p 11434:11434 \
		--memory=6g \
		--memory-swap=8g \
		--cpus=2 \
		ollama/ollama
	@powershell -Command "Write-Host 'Waiting for Ollama to start...'; Start-Sleep -Seconds 30"
	@powershell -Command "Write-Host 'Installing curl in Ollama container...'; docker exec ollama apt-get update; docker exec ollama apt-get install -y curl"
	docker exec ollama ollama pull llama3.2:3b
	@powershell -Command "Write-Host 'Waiting for model to load...'; Start-Sleep -Seconds 30"
