FROM python:3.5

RUN mkdir /tmp-install-talib && cd /tmp-install-talib \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr \
    && make && make install && rm -rf /tmp-install-talib


ENV lambda_home /lambda-trader

WORKDIR ${lambda_home}

ADD ./requirements.txt ${lambda_home}

RUN pip install -r requirements.txt

ADD . ${lambda_home}

CMD /bin/bash
