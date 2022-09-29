BUILD_DIR=build

fmt:
	clang-format -i include/kernel_launcher/*.h src/*.cpp tests/*.cpp examples/*/*.cu

test: ${BUILD_DIR}
	cd ${BUILD_DIR} && make kernel_launcher_tests
	cd tests && KERNEL_LAUNCHER_LOG=debug ../${BUILD_DIR}/tests/kernel_launcher_tests ${TEST}

${BUILD_DIR}:
	mkdir ${BUILD_DIR}
	cd ${BUILD_DIR} && cmake -DKERNEL_LAUNCHER_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=debug ..

all: fmt test
clean:

.PHONY: fmt test all clean
