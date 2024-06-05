CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -O3 -lpthread -std=c++20
CFLAGSDEBUG = -Wall -Wextra -Wpedantic -lpthread -std=c++20 -g

targets: sequential parallel

all: targets
debug: sequential_debug parallel_debug

sequential_debug: sequential.cpp
	$(CC) $(CFLAGSDEBUG) -o sequential sequential.cpp

sequential: sequential.cpp
	$(CC) $(CFLAGS) -o sequential sequential.cpp utils.cpp

parallel: parallel.cpp
	$(CC) $(CFLAGS) -o parallel parallel.cpp utils.cpp

parallel_debug: parallel.cpp
	$(CC) $(CFLAGSDEBUG) -o parallel parallel.cpp utils.cpp

clean:
	rm -f sequential parallel
