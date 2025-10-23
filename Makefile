OPTIMIZED:=-O3
DEBUG_MODE:=

all: debug/wave-simulation
release: release/wave-simulation

debug/wave-simulation: debug/main.o debug/gl.o
	nvcc --std=c++11 -g debug/gl.o debug/main.o -lglfw -o debug/wave-simulation

debug/main.o: src/main.cu
	nvcc --std=c++11 -I include/ -c -g -o debug/main.o src/main.cu

debug/gl.o: src/gl.c
	clang -Wall -Werror -pedantic --std=c17 -I include/ -c -g -o debug/gl.o src/gl.c

release/wave-simulation: release/main.o release/gl.o
	nvcc --std=c++11 $(OPTIMIZED) $(DEBUG_MODE) release/main.o release/gl.o -lglfw -o release/wave-simulation

release/main.o: src/main.cu
	nvcc --std=c++11 -I include/ -c $(OPTIMIZED) $(DEBUG_MODE) -o release/main.o src/main.cu

release/gl.o: src/gl.c
	clang -Wall -Werror -pedantic --std=c17 -I include/ -c $(OPTIMIZED) $(DEBUG_MODE) -o release/gl.o src/gl.c

clean:
	rm -f debug/* release/*
