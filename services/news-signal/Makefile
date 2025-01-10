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

# To generate a golden dataset with tuples to do
# Supervised Fine Tuning
golden-dataset-with-claude:
	uv run python golden-dataset.py \
	 --model-provider anthropic \
	 --n 1000 \
	 --output-file ./data/golden_dataset_anthropic.jsonl

golden-dataset-with-ollama:
	uv run python golden-dataset.py \
	 --model-provider ollama \
	 --n 1000 \
	 --output-file ./data/golden_dataset_ollama.jsonl

fine-tune:
	uv run python fine_tuning.py \
	 --base-llm-name unsloth/Llama-3.2-1B-bnb-4bit \
	 --dataset-path ./data/golden_dataset_ollama.jsonl \
	 --comet-ml-project-name news-signal-extractor
