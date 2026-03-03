FROM rocker/r-ver:4.5

RUN install2.r --error --skipinstalled pak \
    && R -e 'pak::pkg_install("mkanai/fasthurdle")' \
    && rm -rf /tmp/downloaded_packages

CMD ["R"]
