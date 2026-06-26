FROM lukemathwalker/cargo-chef:0.1.77-rust-1.96.0-trixie AS chef
WORKDIR /build

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG CRATE_NAME
SHELL ["/bin/bash", "-c"]
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --profile release --recipe-path recipe.json -p ${CRATE_NAME}
COPY . .
RUN cargo build --locked --release -p ${CRATE_NAME} && \
  mkdir -p /runtime/usr/local/bin && \
  mv "./target/release/${CRATE_NAME}" /runtime/usr/local/bin/server

# TODO: config, logs, https
# c.f. https://github.com/hseeberger/hello-rs/blob/main/Dockerfile
FROM debian:trixie-slim AS runtime
ARG ARTIFACTS_DIR
COPY --from=builder --chown=10001:10001 /runtime /
COPY --chown=10001:10001 ${ARTIFACTS_DIR} /artifacts
USER 10001:10001
EXPOSE 3000
CMD ["server"]

