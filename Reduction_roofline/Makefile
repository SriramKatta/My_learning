CUDA_PATH ?= /usr/local/cuda
NVCC := nvcc

SRC := main.cu
OBJ := main.o
EXEC := mainexe

# Compiler flags
NVCCFLAGS := -std=c++17 -O3 -arch=sm_60 --expt-extended-lambda -I$(CUDA_PATH)/include -Xcompiler -fopenmp  # Change sm_60 to match your GPU architecture

# Default target
all: $(EXEC)

# Compile source files to object files
$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link object files to create the executable
$(EXEC): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Clean build files
clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all clean
