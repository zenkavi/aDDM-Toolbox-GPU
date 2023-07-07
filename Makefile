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
	$(CXX) src/simulate_addm.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/sim_addm
	$(CXX) src/simulate_ddm.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/sim_ddm

nll:
	$(CXX) src/nll_ddm.cpp $(CXXLIBS) $(LIB) $(INC) -o bin/nll_ddm

all: sim main nll
	
run: 
	bin/main
	python3 plots.py