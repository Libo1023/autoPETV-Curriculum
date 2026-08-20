FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# ---- Install Python dependencies ----
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt

# ---- Copy custom trainer into the nnunetv2 package ----
COPY --chown=algorithm:algorithm nnUNetTrainerAutoPETV.py /opt/algorithm/
RUN NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])") && \
    cp /opt/algorithm/nnUNetTrainerAutoPETV.py "$NNUNET_PKG/training/nnUNetTrainer/"

# ---- Copy inference script ----
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# ---- Copy model weights ----
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results

# ---- nnU-Net environment variables ----
RUN mkdir -p /opt/algorithm/nnUNet_raw && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process"]