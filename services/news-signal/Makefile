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

check-ollama:
	@powershell -Command "$$running = docker ps -q -f name=ollama; if (-not $$running) { echo 'Ollama container is not running. Running setup-ollama first...'; $(MAKE) setup-ollama }"
	@powershell -Command "Write-Host 'Testing Ollama connection...'; $$result = docker exec ollama curl -s http://localhost:11434/api/version; if ($$LASTEXITCODE -eq 0) { Write-Host 'Ollama is responding' } else { Write-Host 'Ollama is not responding'; exit 1 }"

run-with-ollama: build check-ollama
	@powershell -Command "Write-Host 'Running network diagnostics...'; docker exec ollama curl -v http://localhost:11434/api/version"
	@powershell -Command "Write-Host 'Checking Ollama container networking...'; docker inspect ollama --format='{{json .NetworkSettings.Networks}}'"
	@powershell -Command "docker run -it --rm \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e MODEL_PROVIDER=ollama \
		-e OLLAMA_HOST=ollama \
		-e OLLAMA_MODEL_NAME=llama3.2:3b \
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
		ollama/ollama
	@powershell -Command "Write-Host 'Waiting for Ollama to start...'; Start-Sleep -Seconds 20"
	@powershell -Command "Write-Host 'Installing curl in Ollama container...'; docker exec ollama apt-get update; docker exec ollama apt-get install -y curl"
	docker exec -it ollama ollama pull llama3.2:3b