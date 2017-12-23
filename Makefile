.PHONY: install-deps
install-deps:
	pip install -r requirements.txt;

.PHONY: install-fabric-deps
install-fabric-deps:
	pip install -r requirements-fabric.txt

.PHONY: rsync-remote-dev
rsync-remote-dev:
	fab -H root@165.227.22.174 rsync_remote_dev

.PHONY: docker-build
docker-build:
	docker build --tag lambdatrader .

.PHONY: docker-compose-build
docker-compose-build:
	docker-compose build

DEBUG_TO_CONSOLE?=False

DAYS?=7
OFFSET?=0

SERVICE?=lambdatrader

.PHONY: run-backtest
run-backtest: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} -e BACKTESTING_NUM_DAYS=${DAYS} -e BACKTESTING_END_OFFSET_DAYS=${OFFSET} ${SERVICE} python3 -m lambdatrader.entrypoints.backtest_driver

.PHONY: run-livetrade
run-livetrade: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} ${SERVICE} python3 -m lambdatrader.entrypoints.livetrade

.PHONY: run-sync-polx-candlesticks
run-sync-polx-candlesticks: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} ${SERVICE} python3 -m lambdatrader.entrypoints.sync_polx_candlesticks

.PHONY: run-mongo-shell
run-mongo-shell: docker-compose-build
	docker-compose exec mongodb mongo

.PHONY: run-apitest
run-apitest: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} ${SERVICE} python3 -m lambdatrader.apitest

.PHONY: run-polx-cancel-all
run-polx-cancel-all: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} ${SERVICE} python3 -m lambdatrader.scripts.polx_cancel_all

.PHONY: tail-info-log
tail-info-log:
	docker-compose exec ${SERVICE} tail -f log/info.log

.PHONY: tail-debug-log
tail-debug-log:
	docker-compose exec ${SERVICE} tail -f log/debug.log

.PHONY: info-log
info-log:
	docker-compose exec ${SERVICE} cat log/info.log

.PHONY: debug-log
debug-log:
	docker-compose exec ${SERVICE} cat log/debug.log

.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down

.PHONY: reset-mongo-volume
reset-mongo-volume:
	docker volume rm lambdatrader_mongodata
