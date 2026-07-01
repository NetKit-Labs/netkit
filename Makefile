# netkit Makefile
#
# Primary build system (GNU Make). See docs/TESTING.md for full test layout.
#
# Common targets:
#   make              — netkit CLI + libnetkit.a (default)
#   make build-all    — netkit + usage examples + C API test binary
#   make test         — C++ regression, then C API regression (36 inference cases)
#   make test-cpp     — ./netkit test
#   make test-c       — ./tests/test_c_api
#   make examples     — infer_cpp + infer_c
#   make export-mnist — regenerate MNIST model + cases (requires numpy)
#   make clean        — remove build products
#   make rebuild      — clean + make

CC = clang
CXX = clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

CORE_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp \
               src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/json_parser.cpp \
               src/model_loader.cpp src/vectors_loader.cpp src/test_mnist.cpp src/netkit_api.cpp \
               src/cli.cpp src/test.cpp
CLI_SOURCES = src/main.cpp

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)
CLI_OBJECTS = $(CLI_SOURCES:.cpp=.o)

EXAMPLE_C = examples/infer_c
EXAMPLE_C_SRC = examples/infer_c.c
EXAMPLE_C_OBJ = examples/infer_c.o

EXAMPLE_CPP = examples/infer_cpp
EXAMPLE_CPP_SRC = examples/infer_cpp.cpp
EXAMPLE_CPP_OBJ = examples/infer_cpp.o

TEST_C = tests/test_c_api
TEST_C_SRC = tests/test_c_api.c
TEST_C_OBJ = tests/test_c_api.o

.PHONY: all lib clean rebuild test test-cpp test-c run example-c example-cpp examples export-mnist export-mnist-cnn export-mnist-all build-all

all: $(TARGET)

build-all: all examples $(TEST_C)

lib: $(LIB)

$(LIB): $(CORE_OBJECTS)
	ar rcs $@ $^

$(TARGET): $(LIB) $(CLI_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB)

$(EXAMPLE_C): $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB)

$(EXAMPLE_CPP): $(LIB) $(EXAMPLE_CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_CPP_OBJ) $(LIB)

$(TEST_C): $(LIB) $(TEST_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_C_OBJ) $(LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_CPP_OBJ): $(EXAMPLE_CPP_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_C_OBJ): $(EXAMPLE_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_C_OBJ): $(TEST_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJECTS) $(CLI_OBJECTS) $(EXAMPLE_C_OBJ) $(EXAMPLE_CPP_OBJ) $(TEST_C_OBJ) \
	      $(TARGET) $(LIB) $(EXAMPLE_C) $(EXAMPLE_CPP) $(TEST_C)

rebuild: clean all

test-cpp: $(TARGET)
	./$(TARGET) test

test-c: $(TEST_C)
	./$(TEST_C)

test: test-cpp test-c

run: test

example-c: $(EXAMPLE_C)

example-cpp: $(EXAMPLE_CPP)

examples: example-cpp example-c

export-mnist:
	python3 tools/export_mnist_mlp.py

export-mnist-cnn:
	python3 tools/export_mnist_cnn.py

export-mnist-all: export-mnist export-mnist-cnn
