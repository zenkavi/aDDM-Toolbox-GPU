CXX := g++
CXXLIBS := lib/ddm.cpp lib/addm.cpp lib/util.cpp
CXXFLAGS := -Ofast -msse4.2 -march=native
LIB := -L lib -lpthread
INC := -I include

NVCC := /usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS := -O3

SRC_DIR := lib
BUILD_DIR := bin 
OBJ_DIR := obj 

sim:
	$(CXX) $(CXXFLAGS) src/ddm_simulate.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_sim
	$(CXX) $(CXXFLAGS) src/addm_simulate.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_sim

nll:
	$(CXX) $(CXXFLAGS) src/ddm_nll.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_nll
	$(CXX) $(CXXFLAGS) src/addm_nll.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_nll
	$(CXX) $(CXXFLAGS) src/ddm_nll_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_nll_thread
	$(CXX) $(CXXFLAGS) src/addm_nll_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_nll_thread

mle: 
	$(CXX) $(CXXFLAGS) src/ddm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_mle
	$(CXX) $(CXXFLAGS) src/addm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_mle
	$(CXX) $(CXXFLAGS) src/ddm_mle_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_mle_thread
	$(CXX) $(CXXFLAGS) src/addm_mle_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_mle_thread

tests:
	$(CXX) $(CXXFLAGS) src/test.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/test

all: sim nll mle tests

gpu: 
	$(NVCC) $(NVCCFLAGS) src/test.cu $(CXXLIBS) $(LIB) $(INC) -o test
	
clean: 
	rm -rf bin/*
	touch bin/.gitkeep