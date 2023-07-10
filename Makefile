CXX := g++
CXXLIBS := lib/ddm.cpp lib/addm.cpp lib/util.cpp
LIB := -L lib
INC := -I include

SRC_DIR := lib
BUILD_DIR := bin 
OBJ_DIR := obj 

main:
	$(CXX) src/main.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/main

sim:
	$(CXX) src/simulate_addm.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_sim
	$(CXX) src/simulate_ddm.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/addm_sim

nll:
	$(CXX) src/ddm_nll.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_nll

mle: 
	$(CXX) src/ddm_mle.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/ddm_mle

all: sim main nll mle
	
clean: 
	rm -rf bin/*
	touch bin/.gitkeep