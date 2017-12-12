rm -rf venv
rm -rf venv-fabric

source scripts/source-make-venv.sh
source scripts/source-make-fabric-venv.sh

source scripts/source-activate-fabric-venv.sh
make install-fabric-deps
deactivate

source scripts/source-activate-venv.sh
make install-deps
