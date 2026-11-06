CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-misleading-indentation
HEADERS := $(wildcard include/hohlraum/*.hpp)
.PHONY: all paper smoke clean
all: build/policy_runner
paper: all build/baseline_runner build/feedback_runner
smoke:
	python3 ../../../../tests/smoke.py
build/policy_runner: src/policy_driver.cpp src/replay.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude $< src/replay.cpp -o $@
build/baseline_runner: src/baseline_driver.cpp $(wildcard src/baselines/*.hpp)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Isrc/baselines $< -o $@
build/feedback_runner: src/admission_feedback.cpp src/admission_pressure.hpp src/replay.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude src/admission_feedback.cpp src/replay.cpp -o $@
clean:
	rm -rf build
