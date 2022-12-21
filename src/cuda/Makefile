CXX := g++
CXXFLAGS := -std=c++17 -O3

NVCC := nvcc
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_80,code=sm_80 -Xcompiler '-fPIC' -g -O3

TEST_APP := test

LIB := libcuda.a

OBJ_DIR := obj

SRC := $(wildcard *.cu)
OBJ := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(SRC))


.PHONY: all clean

all: $(LIB)

$(TEST_APP): $(LIB)
	$(NVCC) test.cpp $(CUDA_LINK_FLAGS) -o $@ $(LIB)

$(LIB): $(OBJ) model.h
	ar -rcs $@ -o $^

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CUDA_COMPILE_FLAGS) $< -c -o $@

clean:
	-rm $(LIB)
	-rm $(OBJ)
