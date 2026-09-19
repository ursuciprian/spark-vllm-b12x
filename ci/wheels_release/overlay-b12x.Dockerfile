# Small overlay used only by build.sh --use-wheels: installs the exact
# released b12x wheel over an already-built runner image, instead of eugr's
# Dockerfile's from-source/from-pypi b12x install (which --use-wheels skips
# by passing B12X_REPO="" to the runner stage).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --from=b12x_wheel /*.whl /tmp/b12x-wheels/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/b12x-wheels/b12x-*.whl && \
    rm -rf /tmp/b12x-wheels
