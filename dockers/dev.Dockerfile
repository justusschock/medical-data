ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

COPY . /tmp/pip-tmp
RUN  python setup.py egg_info --egg-base=/tmp/pip-tmp && pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html -r /tmp/pip-tmp/*.egg-info/requires.txt && rm -rf /tmp/pip-tmp
