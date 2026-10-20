# Fallback build without CMake:   make -j8 && make test
# (CMakeLists.txt is the primary build; this mirrors it for toolchains that only have g++/clang++ + make.)
CXX      ?= g++
OPT      ?= -O2
CXXFLAGS  = $(OPT) -std=c++20 -Wall -Wextra -Iinclude -MMD -MP
BUILD    := build/make

LIB_SRC  := $(shell find src -name '*.cpp')
LIB_OBJ  := $(LIB_SRC:%.cpp=$(BUILD)/%.o)
TEST_SRC := test/unit/FrontendTests.cpp test/unit/PassTests.cpp test/unit/BackendTests.cpp test/unit/TestMain.cpp
TEST_OBJ := $(TEST_SRC:%.cpp=$(BUILD)/%.o)

UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
  LDLIBS += -ldl
endif
ifneq (,$(findstring _NT,$(UNAME)))
  EXE := .exe
  LDFLAGS += -static   # self-contained binary: no MinGW runtime DLLs needed on PATH
endif

all: $(BUILD)/ptx-forge$(EXE) $(BUILD)/ptx-forge-tests$(EXE)

$(BUILD)/libptxforge.a: $(LIB_OBJ)
	@rm -f $@ && ar rcs $@ $^

$(BUILD)/ptx-forge$(EXE): $(BUILD)/tools/ptx-forge-cli.o $(BUILD)/libptxforge.a
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/ptx-forge-tests$(EXE): $(TEST_OBJ) $(BUILD)/libptxforge.a
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test/%.o: test/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Itest/unit -DPTXFORGE_TEST_DIR='"test"' -c $< -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(BUILD)/ptx-forge-tests$(EXE)
	$(BUILD)/ptx-forge-tests$(EXE)

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
-include $(LIB_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
