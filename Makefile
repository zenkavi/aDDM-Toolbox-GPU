NVCC := /usr/local/cuda-12.2/bin/nvcc
CXX := g++

SIM_EXECS := addm_simulate ddm_simulate
NLL_EXECS := addm_nll_thread addm_nll ddm_nll_thread ddm_nll
MLE_EXECS := addm_mle_thread addm_mle ddm_mle_thread ddm_mle
TEST_EXECS := test test_mle_method
GPU_EXECS := addm_nll_gpu ddm_nll_gpu

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
SRC_DIR := src
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

define compile_target_cpp
	$(CXX) $(CXXFLAGS) $(addprefix $(SRC_DIR)/, $1.cpp) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(NVCC) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef

define compile_target_cu
	$(NVCC) $(NVCCFLAGS) $(addprefix $(SRC_DIR)/, $1.cu) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(NVCC) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef

sim: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(foreach source, $(SIM_EXECS), $(call compile_target_cpp, $(source));)


nll: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(foreach source, $(NLL_EXECS), $(call compile_target_cpp, $(source));)


mle: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp, $(source));)


tests: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp, $(source));)


gpu: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(foreach source, $(GPU_EXECS), $(call compile_target_cu, $(source));)


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
