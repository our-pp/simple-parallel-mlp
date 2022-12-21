CXX := g++
CXXFLAGS := -std=c++17 -O3

NVCC := nvcc
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_80,code=sm_80 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_80,code=sm_80 -Xcompiler '-fPIC' -g -O3

OBJ_DIR := obj

EXE_cuda_padding := ../bin/cuda-padding
EXE_singleThread := ../bin/singleThread
EXE_cuda := ../bin/cuda
EXE_openmp := ../bin/openmp

SRC := main.cpp
OBJ := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))

LIB_singleThread := singleThread/libsingleThread.a
LIB_cuda_padding := cuda-padding/libcudapadding.a
LIB_cuda := cuda/libcuda.a
LIB_openmp := openmp/libopenmp.a

.PHONY: all clean

all: $(EXE_singleThread) $(EXE_cuda) $(EXE_openmp) $(EXE_cuda_padding)

$(EXE_singleThread): obj/singleThread.o $(LIB_singleThread)
	$(CXX) $(CXXFLAGS) -IsingleThread -LsingleThread -lsingleThread $^ -o $@

$(EXE_cuda_padding): obj/cuda-padding.o $(LIB_cuda_padding)
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $(LIB_cuda_padding) obj/cuda-padding.o

$(EXE_cuda): obj/cuda.o $(LIB_cuda)
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $(LIB_cuda) obj/cuda.o

$(EXE_openmp): obj/openmp.o $(LIB_openmp)
	$(CXX) $(CXXFLAGS) -Iopenmp -Lopenmp -lopenmp -fopenmp $^ -o $@

obj/singleThread.o: main.cpp $(LIB_singleThread)
	$(CXX) $(CXXFLAGS) -IsingleThread $^ -c -o $@

obj/cuda-padding.o: mainCudaPadding.cpp $(LIB_cuda_padding)
	$(CXX) $(CXXFLAGS) -Icuda-padding $^ -c -o $@

obj/cuda.o: main.cpp $(LIB_cuda)
	$(CXX) $(CXXFLAGS) -Icuda $^ -c -o $@ 

obj/openmp.o: main.cpp $(LIB_openmp)
	$(CXX) $(CXXFLAGS) -Iopenmp -fopenmp $^ -c -o $@

$(LIB_singleThread):
	make -C singleThread

$(LIB_cuda):
	make -C cuda

$(LIB_openmp):
	make -C openmp

$(LIB_cuda_padding):
	make -C cuda-padding

clean:
	-rm obj/*
	-rm $(EXE_cuda)
	-rm $(EXE_openmp)
	-rm $(EXE_cuda_padding)
	-rm $(EXE_singleThread)
	make -C openmp clean
	make -C cuda clean
	make -C cuda-padding clean 
	make -C singleThread clean