CXX := g++
LIB := -L lib
INC := -I include

SRC_DIR := lib
BUILD_DIR := bin 
OBJ_DIR := obj 

all:
	g++ src/main.cpp lib/ddm.cpp lib/addm.cpp lib/util.cpp -L lib -I include -o bin/main

sim:
	$(CXX) src/simulate.cpp lib/ddm.cpp lib/addm.cpp lib/util.cpp $(LIB) $(INC) -o bin/simulate

run: 
	bin/main
	python3 plots.py
