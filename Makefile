NVCC := /usr/local/cuda-12.2/bin/nvcc
CXX := g++

# MACROS := -DEXCLUDE_CUDA_CODE

SIM_EXECS := addm_simulate ddm_simulate
NLL_EXECS := addm_nll_thread addm_nll ddm_nll_thread ddm_nll
MLE_EXECS := addm_mle_thread addm_mle ddm_mle_thread ddm_mle
TEST_EXECS := test benchmark
GPU_EXECS := addm_nll_gpu ddm_nll_gpu

LIB_DIR := lib
OBJ_DIR := obj
INC_DIR := include
BUILD_DIR := bin
SRC_DIR := src

CXXFLAGS := -Ofast -msse4.2 -march=native -fPIC -c $(MACROS)
NVCCFLAGS := -O3 -Xcompiler -fPIC -c $(MACROS)
SHAREDFLAGS = -I $(INC_DIR) -lpthread
LDFLAGS := -shared $(MACROS)
LIB := -L lib -lpthread
INC := -I $(INC_DIR)

INSTALL_LIB_DIR := /usr/lib
INSTALL_INC_DIR := /usr/include

PY_SUFFIX := $(shell python3-config --extension-suffix)
PY_INCLUDES := $(shell python3 -m pybind11 --includes)

CPP_FILES := $(filter-out $(LIB_DIR)/bindings.cpp, $(wildcard $(LIB_DIR)/*.cpp))
CU_FILES := $(wildcard $(LIB_DIR)/*.cu)
CPP_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
ifeq ($(MACROS),)
	CU_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_FILES)) 
endif


$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(SHAREDFLAGS) -o $@ $<

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(SHAREDFLAGS) -o $@ $<

define compile_target_cpp_nvcc
	$(CXX) $(CXXFLAGS) $(addprefix $(SRC_DIR)/, $1.cpp) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(NVCC) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef

define compile_target_cpp_cpu
	$(CXX) $(CXXFLAGS) $(addprefix $(SRC_DIR)/, $1.cpp) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(CXX) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef

define compile_target_cu
	$(NVCC) $(NVCCFLAGS) $(addprefix $(SRC_DIR)/, $1.cu) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(NVCC) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef


sim: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(SIM_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(SIM_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

nll: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(NLL_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(NLL_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

mle: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

test: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(TEST_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(TEST_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

gpu: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(GPU_EXECS), $(call compile_target_cu, $(source));)
endif 

all: sim nll mle test gpu


install: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(NVCC) $(LDFLAGS) -o $(INSTALL_LIB_DIR)/libaddm.so $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
else 
	$(CXX) $(LDFLAGS) -o $(INSTALL_LIB_DIR)/libaddm.so $(CPP_OBJ_FILES)
endif 
	cp -TRv $(INC_DIR) $(INSTALL_INC_DIR)/addm

python: 
	g++ -O3 -shared -fPIC $(PY_INCLUDES) -DEXCLUDE_CUDA_CODE -I include $(CPP_FILES) $(LIB_DIR)/bindings.cpp -o $(addsuffix $(PY_SUFFIX), addm_toolbox_gpu)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(BUILD_DIR)/*
	touch $(BUILD_DIR)/.gitkeep
	touch $(OBJ_DIR)/.gitkeep
	rm -rf docs
