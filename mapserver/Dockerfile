FROM ubuntu:14.04

ENV HARFBUZZ_VERSION=2.4.0 \
    MAPSERVER_VERSION=branch-7-2 \
    APACHE_CONFDIR=/etc/apache2 APACHE_RUN_USER=www-data APACHE_RUN_GROUP=www-data \
    APACHE_RUN_DIR=/var/run/apache2 APACHE_PID_FILE=/var/run/apache2/apache2.pid APACHE_LOCK_DIR=/var/lock/apache2 APACHE_LOG_DIR=/var/log/apache2 \
    LANG=C \
    TERM=linux

# Dependencies
RUN apt-get -qq update --fix-missing && apt-get -qq --yes upgrade

RUN apt-get install -y software-properties-common g++ make cmake wget git bzip2 apache2 apache2-threaded-dev curl apache2-mpm-worker

RUN apt-get install -y libxml2-dev libxslt1-dev libproj-dev libfribidi-dev libcairo2-dev librsvg2-dev libmysqlclient-dev libpq-dev libcurl4-gnutls-dev \
    libexempi-dev libgdal-dev libgeos-dev gdal-bin libfcgi-dev libapache2-mod-fastcgi

# Mapserver and dependency compilation
ADD resources /tmp/resources
ADD setup.sh /setup.sh
RUN chmod 0755 /setup.sh
RUN /setup.sh

# Apache configuration
COPY ./resources/000-default.conf /etc/apache2/sites-available/
RUN echo "ServerName MapserverContainer" >> /etc/apache2/apache2.conf

RUN  a2enmod actions cgi alias && \
    find "${APACHE_CONFDIR}" -type f -exec sed -ri ' \
    s!^(\s*CustomLog)\s+\S+!\1 /proc/self/fd/1!g; \
    s!^(\s*ErrorLog)\s+\S+!\1 /proc/self/fd/2!g; \
    ' '{}' ';'

RUN chmod o+x /usr/local/bin/mapserv
RUN ln -s /usr/local/bin/mapserv /usr/lib/cgi-bin/mapserv.fcgi
RUN chmod 755 /usr/lib/cgi-bin

RUN apt-get clean && \
    rm --recursive --force /var/lib/apt/lists/partial/* /tmp/* /var/tmp/* && \
    adduser ${APACHE_RUN_USER} root && \
    mkdir --parent ${APACHE_RUN_DIR} ${APACHE_LOCK_DIR} ${APACHE_LOG_DIR} && \
    chmod -R g+w /etc/apache2 ${APACHE_RUN_DIR} ${APACHE_LOCK_DIR} ${APACHE_LOG_DIR}

COPY apache2ctl-dforeground /usr/local/bin/
RUN chmod 755 /usr/local/bin/apache2ctl-dforeground
WORKDIR /var/www/html

EXPOSE 80
CMD apache2ctl-dforeground
