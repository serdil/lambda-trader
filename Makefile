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

.PHONY: run-backtest
run-backtest:
	docker-compose run lambdatrader python3 -m lambdatrader.backtest_driver

.PHONY: run-livetrade
run-livetrade:
	docker-compose run lambdatrader python3 -m lambdatrader.livetrade