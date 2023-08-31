all:
	cls && cd build && cmake --build . && .\test\Debug\Test.exe
