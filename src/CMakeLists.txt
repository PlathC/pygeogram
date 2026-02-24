add_subdirectory(extern)

# nanobind setup
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else ()
    set(DEV_MODULE Development.Module)
endif ()

find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

## Detect the installed nanobind package and import it into CMake
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(grampy_ext

        # Target the stable ABI for Python 3.12+, which reduces
        # the number of binary wheels that must be built. This
        # does nothing on older Python versions
        STABLE_ABI

        # Build libnanobind statically and merge it into the
        # extension (which itself remains a shared library)
        #
        # If your project builds multiple extensions, you can
        # replace this flag by NB_SHARED to conserve space by
        # reusing a shared libnanobind across libraries
        NB_STATIC

        grampy_ext.cpp)

get_cxx_flags(GRP_COMPILATION_FLAGS GRP_COMPILE_DEFINITIONS)
target_compile_options(grampy_ext PRIVATE ${GRP_COMPILATION_FLAGS})
target_compile_definitions(grampy_ext PRIVATE ${GRP_COMPILE_DEFINITIONS})
target_include_directories(grampy_ext SYSTEM PRIVATE ${GRP_EXTERN_INCLUDES})
target_include_directories(grampy_ext PRIVATE include/)
target_include_directories(grampy_ext SYSTEM INTERFACE include/)
target_link_libraries(grampy_ext PRIVATE ${GRP_LINK_LIBRARIES})
target_compile_features(grampy_ext PRIVATE cxx_std_20)

# Install directive for scikit-build-core
install(TARGETS grampy_ext LIBRARY DESTINATION grampy)