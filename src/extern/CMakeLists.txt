if (NOT TARGET geogram)
    message(STATUS "Fetching geogram...")
    set(VORPALINE_BUILD_DYNAMIC OFF CACHE BOOL "" FORCE)
    set(GEOGRAM_SUB_BUILD ON CACHE BOOL "" FORCE)
    set(GEOGRAM_LIB_ONLY ON CACHE BOOL "" FORCE)

    # Based on https://github.com/tksuoran/erhe/blob/main/CMakeLists.txt#L93C1-L118C1
    if (CMAKE_SYSTEM_NAME MATCHES Linux)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES x86_64)
            if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                set(VORPALINE_PLATFORM "Linux64-gcc")
            elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(VORPALINE_PLATFORM "Linux64-clang")
            endif ()
        else ()
            if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                set(VORPALINE_PLATFORM "Linux64-nonx86-gcc")
            elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(VORPALINE_PLATFORM "Linux64-nonx86-clang")
            endif ()
        endif ()
    elseif (CMAKE_SYSTEM_NAME MATCHES Darwin)
        set(VORPALINE_PLATFORM "Darwin-clang")
    elseif (CMAKE_SYSTEM_NAME MATCHES Windows)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            set(VORPALINE_PLATFORM "Win64-gcc" CACHE STRING "")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(VORPALINE_PLATFORM "Win-vs-generic" CACHE STRING "")
        endif ()
    elseif (CMAKE_SYSTEM_NAME MATCHES Android)
        set(VORPALINE_PLATFORM "Android-aarch64-gcc")
    endif ()

    add_subdirectory(geogram)
    get_target_property(target_type geogram TYPE)

    if (APPLE)
        target_compile_options(geogram PRIVATE -w)
    endif ()

    set_target_properties(geogram
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}
    )

    set(GRP_LINK_LIBRARIES ${GRP_LINK_LIBRARIES} geogram)
endif ()

set(GRP_LINK_LIBRARIES ${GRP_LINK_LIBRARIES} PARENT_SCOPE)