FROM python:3.10.19-slim-bookworm@sha256:7de0b3589a2b35452fab55d753aa773d59afcbb88220ab118d6604a398cd65e1

ARG ALPHAGEOMETRY_COMMIT=6777cb586cbb46beed28db12dc72c69770b68337
ARG MELIAD_COMMIT=e8af0543441222c1c4c60d58803511f7cf92908b

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/alphageometry/meliad_lib/meliad \
    JAX_PLATFORMS=cpu \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/google-deepmind/alphageometry.git /opt/alphageometry \
    && git -C /opt/alphageometry checkout --detach "$ALPHAGEOMETRY_COMMIT"

WORKDIR /opt/alphageometry
RUN python -m pip install --no-deps --require-hashes \
    --find-links https://storage.googleapis.com/jax-releases/jax_releases.html \
    -r requirements.txt \
    && python -m pip check

RUN sed -i "s/matplotlib.use('TkAgg')/matplotlib.use('Agg')/" /opt/alphageometry/numericals.py \
    && grep -Fq "matplotlib.use('Agg')" /opt/alphageometry/numericals.py \
    && mkdir -p /opt/alphageometry/meliad_lib \
    && git clone https://github.com/google-research/meliad.git /opt/alphageometry/meliad_lib/meliad \
    && git -C /opt/alphageometry/meliad_lib/meliad checkout --detach "$MELIAD_COMMIT"

COPY scripts/r1_verify.sh /usr/local/bin/r1_verify

CMD ["bash"]
