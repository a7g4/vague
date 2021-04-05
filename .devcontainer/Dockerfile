FROM alpine:latest
#FROM ubuntu:latest
run apk add --no-cache build-base cmake ninja zip unzip curl git pkgconfig # clang-extra-tools

ARG VCPKG_PATH=/opt/vcpkg

RUN mkdir -p ${VCPKG_PATH} \
    && git clone https://github.com/Microsoft/vcpkg.git ${VCPKG_PATH} \
    && cd ${VCPKG_PATH} \
    && ./bootstrap-vcpkg.sh -useSystemBinaries

RUN VCPKG_FORCE_SYSTEM_BINARIES=1 ${VCPKG_PATH}/vcpkg install eigen3 magic-enum catch2

ENV CMAKE_TOOLCHAIN_FILE=${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake