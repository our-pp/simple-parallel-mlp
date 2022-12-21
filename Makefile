.PHONY: all clean

all:
	make -C src/singleThread
	make -C src/cuda
	make -C src

clean:
	make -C src clean
	