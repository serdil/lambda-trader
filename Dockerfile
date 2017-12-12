FROM python:3.5

ENV lambda_home /lambda-trader

WORKDIR ${lambda_home}

ADD ./requirements.txt ${lambda_home}

RUN pip install -r requirements.txt

ADD . ${lambda_home}

CMD /bin/bash