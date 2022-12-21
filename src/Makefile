CXX := g++
CXXFLAGS := -std=c++17 -O3

NVCC := nvcc
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_80,code=sm_80 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_80,code=sm_80 -Xcompiler '-fPIC' -g -O3

OBJ_DIR := obj

EXE_singleThread := ../bin/singleThread
EXE_cuda := ../bin/cuda
EXE_openmp := ../bin/openmp

SRC := main.cpp
OBJ := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))

LIB_singleThread := singleThread/libsingleThread.a
LIB_cuda := cuda/libcuda.a
LIB_openmp := openmp/libopenmp.a

.PHONY: all clean

all: $(EXE_singleThread) $(EXE_cuda) $(EXE_openmp)

$(EXE_singleThread): obj/singleThread.o $(LIB_singleThread)
	$(CXX) $(CXXFLAGS) -IsingleThread -LsingleThread -lsingleThread $^ -o $@

$(EXE_cuda): obj/cuda.o $(LIB_cuda)
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $(LIB_cuda) obj/cuda.o

$(EXE_openmp): obj/openmp.o $(LIB_openmp)
	$(CXX) $(CXXFLAGS) -Iopenmp -Lopenmp -lopenmp -fopenmp $^ -o $@

obj/singleThread.o: main.cpp $(LIB_singleThread)
	$(CXX) $(CXXFLAGS) -IsingleThread -LsingleThread -lsingleThread $^ -c -o $@

obj/cuda.o: main.cpp $(LIB_cuda)
	$(CXX) $(CXXFLAGS) -Icuda -Lcuda -lcuda $^ -c -o $@ 

obj/openmp.o: main.cpp $(LIB_openmp)
	$(CXX) $(CXXFLAGS) -Iopenmp -Lopenmp -lopenmp -fopenmp $^ -c -o $@

$(LIB_singleThread):
	make -C singleThread

$(LIB_cuda):
	make -C cuda

$(LIB_openmp):
	make -C openmp

clean:
	-rm obj/*
	-rm $(EXE_openmp)
	-rm $(EXE_cuda)
	-rm $(EXE_singleThread)
	make -C openmp clean
	make -C cuda clean 
	make -C singleThread clean