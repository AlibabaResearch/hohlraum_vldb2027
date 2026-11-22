PYTHON ?= python3
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-misleading-indentation
CORE_CPP := ../../../core/cpp
LIFECYCLE := build/hohlraum/replay_lifecycle_history.hpp
HEADERS := $(wildcard include/hohlraum/*.hpp) $(wildcard $(CORE_CPP)/*.hpp) $(LIFECYCLE)
INCLUDES := -Iinclude -Ibuild -I$(CORE_CPP)

.PHONY: all clean
all: build/policy_runner build/baseline_runner

$(LIFECYCLE): prepare_history.py $(CORE_CPP)/lifecycle_history.hpp
	$(PYTHON) $^ $@

build/policy_runner: src/policy_driver.cpp src/replay.cpp src/trace_input.hpp $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) src/policy_driver.cpp src/replay.cpp $(LDLIBS) -o $@

build/baseline_runner: src/baseline_driver.cpp $(wildcard src/baselines/*.hpp)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Isrc/baselines $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -rf build
