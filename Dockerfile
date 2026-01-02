ARG UBUNTU_VERSION=24.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git cmake

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