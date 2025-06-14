# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-src"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-build"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/tmp"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/src/fff-populate-stamp"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/src"
  "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/src/fff-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/src/fff-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/Tugbars/Desktop/Savitzky-Golay-Filter/build/_deps/fff-subbuild/fff-populate-prefix/src/fff-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
