all:
	g++ main.cpp lib/ddm.cpp lib/addm.cpp lib/util.cpp -L lib -I include -o main

ddm:
	g++ lib/ddm.cpp -L lib -I include
	./a.out
	python3 plots.py

addm:
	g++ lib/addm.cpp -L lib -I include
	./a.out