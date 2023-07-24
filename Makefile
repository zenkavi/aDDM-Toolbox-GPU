MACROS := -DIGNORE_SPACE_CONSTRAINTS

CXX := g++ 
CXXLIBS := lib/ddm.cpp lib/addm.cpp lib/util.cpp
CXXFLAGS := -Ofast -msse4.2 -march=native
LIB := -L lib -lpthread
INC := -I include

NVCC := /usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS := -Xptxas -O3
CUDALIBS := lib/ddm.cu

SRC_DIR := lib
BUILD_DIR := bin
OBJ_DIR := obj

sim:
	$(CXX) $(CXXFLAGS) src/ddm_simulate.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_sim
	$(CXX) $(CXXFLAGS) src/addm_simulate.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/addm_sim

nll:
	$(CXX) $(CXXFLAGS) src/ddm_nll.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_nll $(MACROS)
	$(CXX) $(CXXFLAGS) src/addm_nll.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/addm_nll
	$(CXX) $(CXXFLAGS) src/ddm_nll_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_nll_thread $(MACROS)
	$(CXX) $(CXXFLAGS) src/addm_nll_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/addm_nll_thread

mle: 
	$(CXX) $(CXXFLAGS) src/ddm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_mle
	$(CXX) $(CXXFLAGS) src/addm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/addm_mle
	$(CXX) $(CXXFLAGS) src/ddm_mle_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_mle_thread
	$(CXX) $(CXXFLAGS) src/addm_mle_thread.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/addm_mle_thread

tests:
	$(CXX) $(CXXFLAGS) src/test.cpp $(CXXLIBS) $(LIB) $(INC) -o $(BUILD_DIR)/test

all: sim nll mle tests

gpu: 
	$(NVCC) $(NVCCFLAGS) src/ddm_nll_gpu.cu $(CXXLIBS) $(CUDALIBS) $(LIB) $(INC) -o $(BUILD_DIR)/ddm_nll_gpu
	
clean: 
	rm -rf bin/*
	touch bin/.gitkeep
	rm -rf docs