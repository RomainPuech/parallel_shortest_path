CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -Ofast -flto -funroll-loops -finline-functions -march=native -lpthread -std=c++20
CFLAGSDEBUG = -Wall -Wextra -Wpedantic -lpthread -std=c++20 -g

targets: sequential

all: targets
debug: sequential_debug

sequential_debug: sequential.cpp
	$(CC) $(CFLAGSDEBUG) -o s sequential.cpp utils.cpp

sequential: sequential.cpp
	$(CC) $(CFLAGS) -o s sequential.cpp utils.cpp

clean:
	rm -f s
