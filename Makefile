NVCC := /usr/local/cuda-12.2/bin/nvcc
CXX := g++

CXXFLAGS := -Ofast -msse4.2 -march=native -fPIC -c
NVCCFLAGS := -O3 -Xcompiler -fPIC -c
SHAREDFLAGS = -I include -lpthread
LDFLAGS := -shared
LIB := -L lib -lpthread
INC := -I include

LIB_DIR := lib
OBJ_DIR := obj
INC_DIR := include
BUILD_DIR := bin
INSTALL_LIB_DIR := /usr/lib
INSTALL_INC_DIR := /usr/include

CPP_FILES := $(wildcard $(LIB_DIR)/*.cpp)
CU_FILES := $(wildcard $(LIB_DIR)/*.cu)
CPP_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CU_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_FILES))


$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(SHAREDFLAGS) -o $@ $<


$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(SHAREDFLAGS) -o $@ $<


sim: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -c src/ddm_simulate.cpp $(LIB) $(INC) -o $(OBJ_DIR)/ddm_sim.o
	$(CXX) $(CXXFLAGS) -c src/addm_simulate.cpp $(LIB) $(INC) -o $(OBJ_DIR)/addm_sim.o

	$(NVCC) $(OBJ_DIR)/ddm_sim.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_sim
	$(NVCC) $(OBJ_DIR)/addm_sim.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_sim


nll: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -c src/ddm_nll.cpp $(LIB) $(INC) -o $(OBJ_DIR)/ddm_nll.o
	$(CXX) $(CXXFLAGS) -c src/addm_nll.cpp $(LIB) $(INC) -o $(OBJ_DIR)/addm_nll.o
	$(CXX) $(CXXFLAGS) -c src/ddm_nll_thread.cpp $(LIB) $(INC) -o $(OBJ_DIR)/ddm_nll_thread.o
	$(CXX) $(CXXFLAGS) -c src/addm_nll_thread.cpp $(LIB) $(INC) -o $(OBJ_DIR)/addm_nll_thread.o

	$(NVCC) $(OBJ_DIR)/ddm_nll.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_nll
	$(NVCC) $(OBJ_DIR)/addm_nll.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_nll
	$(NVCC) $(OBJ_DIR)/ddm_nll_thread.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_nll_thread
	$(NVCC) $(OBJ_DIR)/addm_nll_thread.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_nll_thread


mle: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -c src/ddm_mle.cpp $(LIB) $(INC) -o $(OBJ_DIR)/ddm_mle.o
	$(CXX) $(CXXFLAGS) -c src/addm_mle.cpp $(LIB) $(INC) -o $(OBJ_DIR)/addm_mle.o
	$(CXX) $(CXXFLAGS) -c src/ddm_mle_thread.cpp $(LIB) $(INC) -o $(OBJ_DIR)/ddm_mle_thread.o
	$(CXX) $(CXXFLAGS) -c src/addm_mle_thread.cpp $(LIB) $(INC) -o $(OBJ_DIR)/addm_mle_thread.o

	$(NVCC) $(OBJ_DIR)/ddm_mle.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_mle
	$(NVCC) $(OBJ_DIR)/addm_mle.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_mle
	$(NVCC) $(OBJ_DIR)/ddm_mle_thread.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_mle_thread
	$(NVCC) $(OBJ_DIR)/addm_mle_thread.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_mle_thread


tests: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) -c src/test.cpp $(LIB) $(INC) -o $(OBJ_DIR)/test.o
	$(NVCC) $(OBJ_DIR)/test.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/test


gpu: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) -c src/ddm_nll_gpu.cu $(LIB) $(INC) -o $(OBJ_DIR)/ddm_nll_gpu.o
	$(NVCC) $(NVCCFLAGS) -c src/addm_nll_gpu.cu $(LIB) $(INC) -o $(OBJ_DIR)/addm_nll_gpu.o

	$(NVCC) $(OBJ_DIR)/ddm_nll_gpu.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/ddm_nll_gpu
	$(NVCC) $(OBJ_DIR)/addm_nll_gpu.o $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(BUILD_DIR)/addm_nll_gpu


all: sim nll mle tests gpu


install: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(LDFLAGS) -o $(INSTALL_LIB_DIR)/libaddm.so $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	cp -TRv $(INC_DIR) $(INSTALL_INC_DIR)/addm


.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(BUILD_DIR)/*
	touch $(BUILD_DIR)/.gitkeep
	touch $(OBJ_DIR)/.gitkeep
	rm -rf docs
