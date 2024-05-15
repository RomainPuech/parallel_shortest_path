CC = g++
CFLAGS = -Wall -Wextra -pedantic -O3 -lpthread -std=c++11

sequential: sequential.cpp
	$(CC) $(CFLAGS) -o sequential sequential.cpp

clean:
	rm -f sequential