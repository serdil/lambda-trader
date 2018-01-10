from fabric.api import run
from fabric.context_managers import cd
from fabric.contrib.project import rsync_project
from fabric.operations import put
from fabric.state import env

env.use_ssh_config = True


def install_docker():
    put('./scripts/remote/install-docker.sh', './install-docker.sh')
    run('sudo chmod +x ./install-docker.sh')
    run('sudo ./install-docker.sh')
    run('rm ./install-docker.sh')


def put_ssh_key():
    put('./exclude/ssh/id_rsa', '.ssh/id_rsa')
    put('./exclude/ssh/id_rsa.pub', '.ssh/id_rsa.pub')
    run('sudo chmod 400 .ssh/id_rsa')


def clone_repo():
    run('git clone git@github.com:serdil/lambda-trader.git')
    with cd('lambda-trader'):
        run('git checkout stable')


def init_machine():
    put_ssh_key()
    clone_repo()


def rsync_remote_dev():
    exclude_paths = ['db/history.db', 'venv', 'venv-fabric']
    rsync_project(local_dir='./', remote_dir='./lambda-trader/', exclude=exclude_paths)
