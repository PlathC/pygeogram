if (NOT TARGET geogram)
    message(STATUS "Fetching geogram...")
    add_subdirectory(geogram)

    if (APPLE)
        target_compile_options(geogram PRIVATE -w)
    endif ()

    set(GRP_LINK_LIBRARIES ${GRP_LINK_LIBRARIES} geogram)
endif ()

set(GRP_LINK_LIBRARIES ${GRP_LINK_LIBRARIES} PARENT_SCOPE)