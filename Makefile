.PHONY: install-deps
install-deps:
	pip install -r requirements.txt;

.PHONY: install-fabric-deps
install-fabric-deps:
	pip install -r requirements-fabric.txt

.PHONY: docker-build
docker-build:
	docker build --tag lambdatrader .
