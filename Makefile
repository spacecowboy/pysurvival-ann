DEPS = setup.py $(wildcard src/*.cpp) $(wildcard src/*.h*) $(wildcard ann/*.py)

test: test.py inplace $(DEPS) valgrind
	nosetests -v -x -s test.py
	#nosetests -v ./

nose: inplace $(DEPS)
	nosetests -v -x -s test.py

perf: test.py ann/_ann.so $(DEPS)
	python -m cProfile -s cumulative test.py

specialtest: test.py ann/_ann.so $(DEPS)
	python test.py

valgrind:
	cd src;make -j4 valgrind

inplace: ann/_ann.so

ann/__init__.py:
	#CC=clang++ python setup.py build_ext --inplace
	python setup.py build_ext --inplace

ann/_ann.so: $(DEPS)
	python setup.py build_ext --inplace

install: $(DEPS)
	#python setup.py install
	pip install -r requirements.txt
	pip install -e .

build: $(DEPS)
	python setup.py build

clean:
	rm -f ann/*.so
	rm -rf build/
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf ann/__pycache__
	rm -rf ann/*.pyc
