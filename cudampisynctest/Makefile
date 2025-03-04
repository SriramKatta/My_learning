# Variables
NVCC ?= nvcc
CFLAGS =  -forward-unknown-to-host-compiler -O3 -Wall 
CUFLAGS = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80
SRC = test.cu
OBJ_DIR = build
BIN_DIR = EXE
TARGET = $(BIN_DIR)/test


INCLUDES  = -I$(OPENMPI_ROOT)/include
LIBRARIES = -L$(OPENMPI_ROOT)/lib -lmpi


# Default target
all: $(OBJ_DIR) $(BIN_DIR) $(TARGET)

# Rule to create build and bin directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build the object file
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(CUFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to link the executable
$(TARGET): $(OBJ_DIR)/test.o | $(BIN_DIR)
	$(NVCC) $(CUFLAGS) $(CFLAGS) $(LIBRARIES) -o $@ $<

# Clean up build artifacts
clean: cleanobj cleanexe

cleanobj:
	rm -rf $(OBJ_DIR) 

cleanexe:
	rm -rf $(BIN_DIR)


# Phony targets
.PHONY: all clean cleanobj cleanexe
