.PHONY: install-deps
install-deps:
	pip install -r requirements.txt;

.PHONY: install-fabric-deps
install-fabric-deps:
	pip install -r requirements-fabric.txt
