BUILD_DIR=build

pretty:
	clang-format -i include/*.h include/*/*.h include/*/*/*.h src/*.cpp src/*/*.cpp tests/*.cpp examples/*/*.cu

fmt: pretty

test: ${BUILD_DIR}
	cd ${BUILD_DIR} && make kernel_launcher_tests
	cd tests && KERNEL_LAUNCHER_LOG=debug ../${BUILD_DIR}/tests/kernel_launcher_tests ${TEST}

${BUILD_DIR}:
	mkdir ${BUILD_DIR}
	cd ${BUILD_DIR} && cmake -DKERNEL_LAUNCHER_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=debug ..

all: pretty test
clean:

.PHONY: pretty fmt test all clean
