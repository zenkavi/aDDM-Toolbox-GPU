ddm:
	g++ lib/ddm.cpp -L lib -I include
	./a.out
	python3 plots.py

addm:
	g++ lib/addm.cpp -L lib -I include
	./a.out