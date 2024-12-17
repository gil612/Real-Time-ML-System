# Redpanda management
start-redpanda: 
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

add-one-partition-to-trades-topic:
	docker compose -f redpanda.yml exec redpanda rpk topic add-partitions trades --num 1

# Technical indicators pipeline management
# for live data
build-technical-indicators-live:
	docker compose -f technical-indicators-live.yml build

start-technical-indicators-live: build-technical-indicators-live
	docker compose -f technical-indicators-live.yml up -d

stop-technical-indicators-live:
	docker compose -f technical-indicators-live.yml down

# for historical data
build-technical-indicators-historical:
	docker compose -f technical-indicators-historical.yml build
	
backfill-technical-indicators: build-technical-indicators-historical
	$(eval RUN_ID := $(shell echo %RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%))
	@echo "Starting backfill with RUN_ID: $(RUN_ID)"
	set "RUN_ID=$(RUN_ID)" && docker compose -f technical-indicators-historical.yml up -d

get-random-run-id:
	$(eval RUN_ID := $(shell echo %RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%))
	@echo "Generated RUN_ID: $(RUN_ID)"

clean-backfill-kakfa-topics:
	@echo "Removing all Kafka topics that contain the word historical"
	