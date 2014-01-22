DEPS = setup.py $(wildcard src/*.cpp) $(wildcard src/*.h*) $(wildcard ann/*.py)

perf: test.py ann/_ann.so $(DEPS)
	python -m cProfile -s cumulative test.py

test: test.py ann/_ann.so $(DEPS)
	nosetests -v -x -s test.py

inplace: ann/_ann.so

ann/__init__.py:
	CC=clang++ python setup.py build_ext --inplace

ann/_ann.so: $(DEPS)
	CC=clang++ python setup.py build_ext --inplace

install: $(DEPS)
	#CC=clang++ python setup.py install
	CC=clang++ pip install -r requirements.txt
	CC=clang++ pip install -e .

build: $(DEPS)
	CC=clang++ python setup.py build

clean:
	rm -f ann/*.so
	rm -rf build/
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf ann/__pycache__
	rm -rf ann/*.pyc
