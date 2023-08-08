FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && \
    # apt-get upgrade -y && \
    apt-get wget -y

RUN wget -O /usr/include/c++/11/BS_thread_pool.hpp https://raw.githubusercontent.com/bshoshany/thread-pool/master/include/BS_thread_pool.hpp && \
    mkdir /usr/include/c++/11/nlohmann/ && \
    wget -O /usr/include/c++/11/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp && \
    apt-get install -y libboost-math-dev libboost-math1.74-dev && \
    apt-get install -y git && \
    apt-get install vim


RUN cd home  && \
    git clone https://github.com/jakegoldm/aDDM-Toolbox-GPU.git && \
    cd aDDM-Toolbox-GPU/ && \
    make install


