# The Linux build (CMakeLists.txt includes this file instead of its clang-cl check): toolchain-linux.cmake's clang
# 23.1.2 and ld.lld, configured through build_linux.sh, with every Windows option in its GNU-driver spelling and the
# same intent. Outputs keep their Windows names and directories with Linux suffixes: chainner_native.so,
# _chainner_graph.so and chainner_ext.so.
foreach(chainner_c_language C CXX)
    if(NOT CMAKE_${chainner_c_language}_COMPILER_ID STREQUAL "Clang"
        OR NOT CMAKE_${chainner_c_language}_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
        OR NOT CMAKE_${chainner_c_language}_COMPILER_VERSION VERSION_EQUAL 23.1.2)
        message(FATAL_ERROR "chaiNNer-C builds on Linux with clang 23.1.2 only, not "
            "${CMAKE_${chainner_c_language}_COMPILER_ID} ${CMAKE_${chainner_c_language}_COMPILER_VERSION} "
            "(${CMAKE_${chainner_c_language}_COMPILER_FRONTEND_VARIANT}); configure through native/build_linux.sh")
    endif()
endforeach()
if(CHAINNER_C_FUZZ)
    message(FATAL_ERROR "CHAINNER_C_FUZZ builds are Windows-only (tests/fuzz/README.md)")
endif()

# CMakeLists.txt states each option once, in clang-cl's spelling (ThinLTO, /fp:strict, dynamic denormals, the macro
# prefix map, the ISA units' /clang: flags, /W3, /W4, /WX, /EHsc, /Brepro, and the PDB build's /Zi and linker
# options); chainner_c_gnu_targets rewrites them on every target and source once the whole tree is configured.
# What clang-cl gives every Windows target by default, the GNU driver needs said:
# - position-independent code, which every object of a shared object needs (a DLL's are relocated instead);
# - hidden visibility: a DLL exports only its dllexport symbols, so only CN_EXPORT (and Python's PyInit_*) leave a
#   module; with --exclude-libs the static libraries' symbols (the vendored libraries' default-visibility APIs)
#   stay inside the module that links them, as their copies inside a DLL do;
# - /GS: -fstack-protector-strong;
# - sqrt and sqrtf as the static UCRT computes them (src/ucrt_sqrt.c): clang-cl's /fp:strict keeps math errno, so
#   every Windows module calls the UCRT's out of line; here -fno-builtin-sqrt and -fno-builtin-sqrtf make every call
#   (libstdc++'s std::sqrt included) reach chainner_ucrt_sqrt, which chainner_c_link_targets links into every
#   executable and shared object;
# - libstdc++ and libgcc linked statically, as the static MSVC runtime is (CMAKE_MSVC_RUNTIME_LIBRARY), so the
#   modules need only glibc;
# - load-time binding of every import into read-only memory, as a DLL's import table: -z now and -z relro;
# - POSIX threads (libpthread before glibc 2.34).
# Reproducible output needs nothing for /Brepro: ELF objects and lld's output carry no timestamps.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
add_compile_options(-fstack-protector-strong -fno-builtin-sqrt -fno-builtin-sqrtf -pthread)
add_link_options(-flto=thin $<$<LINK_LANGUAGE:CXX>:-static-libstdc++> -static-libgcc -pthread -Wl,--exclude-libs,ALL
    -Wl,-z,now -Wl,-z,relro)
# Run paths: every target links with its INSTALL_RPATH, empty unless chainner_c_link_targets gives one, so no build
# machine path reaches a binary (CMake's build run path holds absolute directories for outputs outside the build
# tree, as backend/src is).
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
# Compiled outside ThinLTO, so the optimizer never sees a definition of a C library function.
add_library(chainner_ucrt_sqrt OBJECT src/ucrt_sqrt.c)
set_target_properties(chainner_ucrt_sqrt PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_compile_options(chainner_ucrt_sqrt PRIVATE -Wall -Wextra -Werror)

# The GNU spelling of one clang-cl option (a list, empty when the option has no Linux counterpart). An option this
# function does not know stops the configuration instead of reaching the compiler unchecked.
function(chainner_c_gnu_option option out)
    if(option MATCHES "^/clang:(.*)$")
        # -ffile-prefix-map is -fmacro-prefix-map plus the debug information's map.
        string(REGEX REPLACE "^-fmacro-prefix-map=" "-ffile-prefix-map=" value "${CMAKE_MATCH_1}")
    elseif(option STREQUAL "/fp:strict")
        set(value -ffp-model=strict)
    elseif(option STREQUAL "/W3")
        set(value -Wall)
    elseif(option STREQUAL "/W4")
        set(value -Wall -Wextra)
    elseif(option STREQUAL "/WX")
        set(value -Werror)
    elseif(option STREQUAL "/Zi")
        set(value -g)
    elseif(option STREQUAL "/OPT:REF")
        set(value -Wl,--gc-sections)
    elseif(option STREQUAL "/OPT:ICF")
        set(value -Wl,--icf=all)
    elseif(option MATCHES "^/MAP:(.+)$")
        set(value "-Wl,-Map=${CMAKE_MATCH_1}")
    elseif(option MATCHES "^/(EHsc|Brepro|DEBUG)$")
        # C++ exceptions are the GNU driver's default; /DEBUG's information comes from -g.
        set(value "")
    elseif(option MATCHES "^/")
        message(FATAL_ERROR "No Linux spelling for the clang-cl option '${option}' (native/cmake/linux.cmake)")
    else()
        set(value "${option}")
    endif()
    set(${out} "${value}" PARENT_SCOPE)
endfunction()

# Rewrites the option list in the variable list_var, including an option inside a generator expression
# ($<condition:/option>).
function(chainner_c_gnu_options list_var)
    set(result "")
    foreach(option IN LISTS ${list_var})
        if(option MATCHES "^(\\$<.+:)(/[^>]+)>$")
            set(prefix "${CMAKE_MATCH_1}")
            chainner_c_gnu_option("${CMAKE_MATCH_2}" mapped)
            foreach(item IN LISTS mapped)
                list(APPEND result "${prefix}${item}>")
            endforeach()
        else()
            chainner_c_gnu_option("${option}" mapped)
            list(APPEND result ${mapped})
        endif()
    endforeach()
    set(${list_var} "${result}" PARENT_SCOPE)
endfunction()

# Rewrites the compile and link options of every target of directory and its subdirectories, and the compile options
# of their sources (a source shared by two targets is rewritten twice, which leaves GNU spellings as they are).
function(chainner_c_gnu_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        foreach(property COMPILE_OPTIONS LINK_OPTIONS)
            get_target_property(options ${target} ${property})
            if(options)
                chainner_c_gnu_options(options)
                set_property(TARGET ${target} PROPERTY ${property} "${options}")
            endif()
        endforeach()
        get_target_property(sources ${target} SOURCES)
        get_target_property(source_directory ${target} SOURCE_DIR)
        foreach(source IN LISTS sources)
            if(source MATCHES "^\\$<")
                continue()
            endif()
            cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${source_directory}" NORMALIZE)
            get_source_file_property(options "${source}" TARGET_DIRECTORY ${target} COMPILE_OPTIONS)
            if(options)
                chainner_c_gnu_options(options)
                set_source_files_properties("${source}" TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS
                    "${options}")
            endif()
        endforeach()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        chainner_c_gnu_targets("${subdirectory}")
    endforeach()
endfunction()

# The directory a target's file goes to, absolute.
function(chainner_c_output_directory target out)
    get_target_property(type ${target} TYPE)
    if(type STREQUAL "EXECUTABLE")
        get_target_property(directory ${target} RUNTIME_OUTPUT_DIRECTORY)
    else()
        get_target_property(directory ${target} LIBRARY_OUTPUT_DIRECTORY)
    endif()
    get_target_property(binary_directory ${target} BINARY_DIR)
    if(NOT directory)
        set(directory "${binary_directory}")
    endif()
    cmake_path(ABSOLUTE_PATH directory BASE_DIRECTORY "${binary_directory}" NORMALIZE)
    set(${out} "${directory}" PARENT_SCOPE)
endfunction()

# Every executable and shared object of directory and its subdirectories links chainner_ucrt_sqrt; one that links
# chainner_native.so finds it as a DLL beside it is found: RUNPATH $ORIGIN plus the relative path from its directory
# to chainner_native.so's ($ORIGIN for _chainner_graph.so, $ORIGIN/../nodes/impl for chainner_ext.so).
function(chainner_c_link_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(NOT type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
            continue()
        endif()
        target_link_libraries(${target} PRIVATE chainner_ucrt_sqrt)
        get_target_property(libraries ${target} LINK_LIBRARIES)
        if(NOT "chainner_native" IN_LIST libraries)
            continue()
        endif()
        chainner_c_output_directory(chainner_native native_directory)
        chainner_c_output_directory(${target} target_directory)
        cmake_path(RELATIVE_PATH native_directory BASE_DIRECTORY "${target_directory}" OUTPUT_VARIABLE relative)
        string(REGEX REPLACE "/$" "" relative "${relative}")
        if(relative STREQUAL "" OR relative STREQUAL ".")
            set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
        else()
            set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/${relative}")
        endif()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        chainner_c_link_targets("${subdirectory}")
    endforeach()
endfunction()

# At the end of CMakeLists.txt's processing, also after its early return() when the TensorRT parts are skipped.
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL chainner_c_gnu_targets "${CMAKE_CURRENT_SOURCE_DIR}")
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL chainner_c_link_targets "${CMAKE_CURRENT_SOURCE_DIR}")
