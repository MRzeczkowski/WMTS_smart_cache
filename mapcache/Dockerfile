FROM ubuntu:14.04

ENV MAPCACHE_VERSION=branch-1-8 \
    APACHE_CONFDIR=/etc/apache2 APACHE_RUN_USER=www-data APACHE_RUN_GROUP=www-data \
    APACHE_RUN_DIR=/var/run/apache2 APACHE_PID_FILE=/var/run/apache2/apache2.pid APACHE_LOCK_DIR=/var/lock/apache2 APACHE_LOG_DIR=/var/log/apache2 \
    LANG=C \
    TERM=linux

# Dependencies
RUN apt-get -qq update --fix-missing && apt-get -qq --yes upgrade

RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends ca-certificates git cmake build-essential \
    liblz-dev libpng-dev libgdal-dev libgeos-dev libsqlite0-dev libpixman-1-dev libcurl4-openssl-dev libaprutil1-dev libapr1-dev libjpeg-dev \
    libdpkg-dev libdb5.3-dev libtiff5-dev libpcre3-dev \
    apache2 apache2-dev

# Mapserver compilation
RUN git clone --single-branch --branch ${MAPCACHE_VERSION} https://github.com/mapserver/mapcache.git /tmp/resources/mapcache; \
    mkdir -p /tmp/resources/mapcache/build && \
    cd /tmp/resources/mapcache/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DWITH_MEMCACHE=1 -DWITH_FCGI=0 -DWITH_CGI=0 .. && \
    make && \
    make install && \
    ldconfig

# Apache configuration
COPY mapcache.load /etc/apache2/mods-available/
RUN echo "ServerName MapcacheContainer" >> /etc/apache2/apache2.conf

RUN a2enmod mapcache rewrite && \
    find "${APACHE_CONFDIR}" -type f -exec sed -ri ' \
    s!^(\s*CustomLog)\s+\S+!\1 /proc/self/fd/1!g; \
    s!^(\s*ErrorLog)\s+\S+!\1 /proc/self/fd/2!g; \
    ' '{}' ';'

RUN apt-get clean && \
    rm --recursive --force /var/lib/apt/lists/partial/* /tmp/* /var/tmp/* && \
    adduser www-data root && \
    mkdir --parent ${APACHE_RUN_DIR} ${APACHE_LOCK_DIR} ${APACHE_LOG_DIR} && \
    chmod -R g+w /etc/apache2 ${APACHE_RUN_DIR} ${APACHE_LOCK_DIR} ${APACHE_LOG_DIR}

COPY apache2ctl-dforeground /usr/local/bin/
RUN chmod 755 /usr/local/bin/apache2ctl-dforeground
RUN mkdir -p -m 755 /var/sig/tiles/
RUN chown www-data /var/sig/tiles/

WORKDIR /var/www/html

EXPOSE 80
CMD apache2ctl-dforeground

# Copy services definition and create symbolic links for Apache
COPY services /gp_cache/
RUN find /gp_cache -name "*.conf" -exec ln -s {} "/etc/apache2/conf-enabled/" \;

RUN /etc/init.d/apache2 restart