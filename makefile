NVCC	:=nvcc --cudart=static -ccbin g++
CFLAGS	:=-O3 -std=c++11 -rdc=true#--use_fast_math

INC_DIR	:=-I/usr/local/cuda-10.0/samples/common/inc -Icub/
LIB_DIR	:=
LIBS	:= -lcublas -lcudart -lcudadevrt

ARCHES :=-gencode arch=compute_72,code=\"compute_72,sm_72\" \
	-gencode arch=compute_70,code=\"compute_70,sm_70\"

SOURCES :=sdot

all: $(SOURCES)
.PHONY: all

sdot: sdot.cu
	/usr/local/cuda-10.0/bin/$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

clean:
	rm -f $(SOURCES)
