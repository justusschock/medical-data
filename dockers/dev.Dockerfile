ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

COPY . /tmp/pip-tmp
RUN  cd /tmp/pip-tmp && python /tmp/pip-tmp/setup.py egg_info && \
  pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html -r /tmp/pip-tmp/*.egg-info/requires.txt && \
  cd && \
  rm -rf /tmp/pip-tmp
