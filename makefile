CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -O3 -lpthread -std=c++20

targets: sequential parallel

all: targets

sequential: sequential.cpp
	$(CC) $(CFLAGS) -o sequential sequential.cpp

parallel: parallel.cpp
	$(CC) $(CFLAGS) -o parallel parallel.cpp utils.cpp

clean:
	rm -f sequential parallel
