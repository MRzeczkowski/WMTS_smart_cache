wget https://github.com/harfbuzz/harfbuzz/releases/download/${HARFBUZZ_VERSION}/harfbuzz-${HARFBUZZ_VERSION}.tar.bz2 -P /tmp/resources/; \
cd /tmp/resources &&\
tar xjf harfbuzz-${HARFBUZZ_VERSION}.tar.bz2  &&\
cd harfbuzz-${HARFBUZZ_VERSION} && \
./configure  && \
make  && \
make install  && \
ldconfig

git clone --single-branch --branch ${MAPSERVER_VERSION} https://github.com/mapserver/mapserver.git /tmp/resources/mapserver; \
mkdir -p /tmp/resources/mapserver/build && \
cd /tmp/resources/mapserver/build && \
CFLAGS="-std=c99" \
cmake /tmp/resources/mapserver/ \
    -DWITH_THREAD_SAFETY=1 \
    -DWITH_PROJ=1 \
    -DWITH_KML=1 \
    -DWITH_SOS=1 \
    -DWITH_WMS=1 \
    -DWITH_FRIBIDI=1 \
    -DWITH_HARFBUZZ=1 \
    -DWITH_ICONV=1 \
    -DWITH_CAIRO=1 \
    -DWITH_RSVG=1 \
    -DWITH_MYSQL=0 \
    -DWITH_GEOS=1 \
    -DWITH_POSTGIS=0 \
    -DWITH_GDAL=1 \
    -DWITH_OGR=1 \
    -DWITH_CURL=1 \
    -DWITH_CLIENT_WMS=1 \
    -DWITH_CLIENT_WFS=0 \
    -DWITH_WFS=0 \
    -DWITH_WCS=0 \
    -DWITH_LIBXML2=1 \
    -DWITH_GIF=1 \
    -DWITH_EXEMPI=1 \
    -DWITH_XMLMAPFILE=1 \
    -DWITH_PROTOBUFC=0 \
    -DWITH_FCGI=1 && \
make && \
make install && \
ldconfig
