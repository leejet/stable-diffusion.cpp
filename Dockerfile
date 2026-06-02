ARG UBUNTU_VERSION=24.04

FROM ubuntu:$UBUNTU_VERSION AS build

# sd-server embeds the web UI at build time, so the build image needs Node/pnpm.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git cmake ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key -o /tmp/nodesource-repo.gpg.key && \
    gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg /tmp/nodesource-repo.gpg.key && \
    rm /tmp/nodesource-repo.gpg.key && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g pnpm@10.15.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sd.cpp

COPY . .

RUN cmake . -B ./build
RUN cmake --build ./build --config Release --parallel

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends libgomp1 && \
    apt-get clean

COPY --from=build /sd.cpp/build/bin/sd-cli /sd-cli
COPY --from=build /sd.cpp/build/bin/sd-server /sd-server

ENTRYPOINT [ "/sd-cli" ]
