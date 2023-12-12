FROM busybox as downloader
WORKDIR /home
RUN wget -O- https://github.moeyy.xyz/https://github.com/openvpi/SOME/releases/latest/download/0918_continuous256_clean_3spk_fixmel.zip|unzip -
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
COPY . /opt/app
WORKDIR /opt/app
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt gradio==3.47.1
COPY --from=downloader /home experiments
EXPOSE 7860
CMD [ "python", "webui.py", "--addr=0.0.0.0" ]
