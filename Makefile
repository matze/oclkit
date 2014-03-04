all: cmake
	@cd build && make

cmake: clean
	@mkdir -p build
	@cd build && cmake ..

clean:
	@rm -rf build/
