from fabric.api import run
from fabric.context_managers import cd
from fabric.contrib.project import rsync_project
from fabric.operations import put


def install_dtach():
    run('apt install dtach')


def install_pip3():
    run('apt install python3-pip')


def install_programs():
    install_dtach()
    install_pip3()


def install_docker():
    put('./scripts/remote/install-docker.sh', './install-docker.sh')
    run('chmod +x ./install-docker.sh')
    run('./install-docker.sh')
    run('rm ./install-docker.sh')


def put_ssh_key():
    put('./exclude/ssh/id_rsa', '.ssh/id_rsa')
    put('./exclude/ssh/id_rsa.pub', '.ssh/id_rsa.pub')
    run('chmod 400 .ssh/id_rsa')


def runbg(cmd, sockname="dtach"):
    return run('dtach -n `mktemp -u /tmp/%s.XXXX` %s'  % (sockname,cmd))


def clone_repo():
    run('git clone git@github.com:serdil/lambda-trader.git')
    with cd('lambda-trader'):
        run('git checkout stable')


def install_requirements():
    with cd('lambda-trader'):
        run('pip3 install -r requirements.txt')


def init_machine():
    install_programs()
    put_ssh_key()
    clone_repo()
    install_requirements()


def put_config(config_path='./lambdatrader/config.py'):
    put(config_path, 'lambda-trader/lambdatrader/config.py')


def start_bot():
    with cd('lambda-trader'):
        runbg('python3 -m lambdatrader.livetrade')


def stop_bot():
    with cd('lambda-trader'):
        run('killall python3 || true')


def update_bot():
    with cd('lambda-trader'):
        run('git pull origin stable')


def set_up_remote_dev():
    install_programs()
    install_docker()


def rsync_remote_dev():
    rsync_project(remote_dir='./lambda-trader/')