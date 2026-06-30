CC = clang
CXX = clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

CORE_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp \
               src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/json_parser.cpp \
               src/model_loader.cpp src/vectors_loader.cpp src/netkit_api.cpp
CLI_SOURCES = src/main.cpp src/cli.cpp src/test.cpp

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)
CLI_OBJECTS = $(CLI_SOURCES:.cpp=.o)

EXAMPLE_C = examples/infer_c
EXAMPLE_C_SRC = examples/infer_c.c
EXAMPLE_C_OBJ = examples/infer_c.o

all: $(TARGET)

lib: $(LIB)

$(LIB): $(CORE_OBJECTS)
	ar rcs $@ $^

$(TARGET): $(LIB) $(CLI_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB)

$(EXAMPLE_C): $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_C_OBJ): $(EXAMPLE_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJECTS) $(CLI_OBJECTS) $(EXAMPLE_C_OBJ) $(TARGET) $(LIB) $(EXAMPLE_C)

rebuild: clean all

run: $(TARGET)
	./$(TARGET) test

example-c: $(EXAMPLE_C)

.PHONY: all lib clean rebuild run example-c
