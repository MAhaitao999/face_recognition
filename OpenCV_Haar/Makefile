SOURCE = $(wildcard *.cpp)
TARGETS = $(patsubst %.cpp, %, $(SOURCE))

CC = g++
CPPFLAGS = -std=c++11 -Wall -g `pkg-config --cflags --libs opencv`

all: $(TARGETS)

$(TARGETS):%:%.cpp
	$(CC) $< $(CPPFLAGS) -o $@

.PHONY: clean all

clean:
	rm -rf $(TARGETS)
