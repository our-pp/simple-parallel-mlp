CXX := g++
CXXFLAGS := -std=c++17 -O3 -fopenmp

LIB := libopenmp.a

OBJ_DIR := obj

SRC := $(wildcard *.cpp)
OBJ := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))


.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJ) model.h
	ar -rcs $@ -o $^

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OBJFLAGS) $< -c -o $@

clean:
	-rm $(LIB)
	-rm $(OBJ)
