#define CATCH_CONFIG_RUNNER
#include <filesystem>

#include "catch.hpp"

int main(int argc, char* argv[]) {
    // Tests refer to files relative to the tests directory (e.g., "assets/..."), so run from
    // there regardless of where the binary is invoked.
    std::filesystem::current_path(KERNEL_LAUNCHER_TESTS_DIR);

    return Catch::Session().run(argc, argv);
}
