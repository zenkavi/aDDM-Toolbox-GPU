CXX := g++
CXXLIBS := lib/ddm.cpp lib/addm.cpp lib/util.cpp
CXXFLAGS := -O3
LIB := -L lib -lpthread
INC := -I include

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

mle: 
	$(CXX) $(CXXFLAGS) src/ddm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_mle
	$(CXX) $(CXXFLAGS) src/addm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_mle

tests:
	$(CXX) $(CXXFLAGS) src/test.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/test

all: sim nll mle tests

gpu: 
	nvcc -O3 src/test.cu -lcublas $(CXXLIBS) $(LIB) $(INC) -o test
	
clean: 
	rm -rf bin/*
	touch bin/.gitkeep