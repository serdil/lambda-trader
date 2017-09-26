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

.PHONY: run-backtest docker-compose-build
run-backtest:
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} lambdatrader python3 -m lambdatrader.backtest_driver

.PHONY: run-livetrade
run-livetrade: docker-compose-build
	docker-compose run -e DEBUG_TO_CONSOLE=${DEBUG_TO_CONSOLE} lambdatrader python3 -m lambdatrader.livetrade

.PHONY: tail-info-log
tail-info-log:
	docker-compose exec lambdatrader tail -f log/info.log

.PHONY: tail-debug-log
tail-debug-log:
	docker-compose exec lambdatrader tail -f log/debug.log

.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down

.PHONY: reset-mongo-volume
reset-mongo-volume:
	docker volume rm lambdatrader_mongodata