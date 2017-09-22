FROM python:3.5

ENV lambda_home /lambda-trader

WORKDIR ${lambda_home}

ADD . ${lambda_home}

RUN pip install -r requirements.txt

CMD /bin/bash