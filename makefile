CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -O3 -lpthread -std=c++20
CFLAGSDEBUG = -Wall -Wextra -Wpedantic -lpthread -std=c++20 -g

targets: main

all: targets

debug: main.cpp
	$(CC) $(CFLAGSDEBUG) -o main main.cpp utils.cpp

main: main.cpp
	$(CC) $(CFLAGS) -o main main.cpp utils.cpp

clean:
	rm -f s
