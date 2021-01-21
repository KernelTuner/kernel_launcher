CFLAGS=-std=c++11 -g -Wall -Wextra -I../include -I../thirdparty/nlohmann/json/single_include

example: example.cpp ../include/kernel_launcher.h
	g++ example.cpp -o example $(CFLAGS) -lnvrtc -lcuda

clean:
	rm example
