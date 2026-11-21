# The models a build contains. Adding a model is one word in HALO_BUILTIN_MODELS plus its own library under
# src/models/<arch> (named model_<arch>, in the namespace halo::models::<arch>, with make_adapter()), gated by
# `if(HALO_MODEL_<ARCH>)`. A model left out is not compiled, not linked and not registered.

# Every model that ships: the architecture ids, as `general.architecture` names them.
set(HALO_BUILTIN_MODELS qwen4exp qwen35 inkling2)

set(HALO_MODELS "${HALO_BUILTIN_MODELS}" CACHE STRING
    "Semicolon list of the models to build, from: ${HALO_BUILTIN_MODELS}")

list(REMOVE_DUPLICATES HALO_MODELS)
string(REPLACE ";" ", " _halo_valid "${HALO_BUILTIN_MODELS}")
if(NOT HALO_MODELS)
  message(FATAL_ERROR "HALO_MODELS is empty: name at least one of ${_halo_valid}")
endif()
foreach(_m IN LISTS HALO_MODELS)
  if(NOT _m IN_LIST HALO_BUILTIN_MODELS)
    message(FATAL_ERROR "HALO_MODELS names '${_m}', which is not a Halocline model. Valid: ${_halo_valid}")
  endif()
endforeach()

foreach(_m IN LISTS HALO_BUILTIN_MODELS)
  string(TOUPPER ${_m} _up)
  if(_m IN_LIST HALO_MODELS)
    set(HALO_MODEL_${_up} ON)
  else()
    set(HALO_MODEL_${_up} OFF)
  endif()
endforeach()
message(STATUS "halo: models ${HALO_MODELS}")

# The registration list the composition root (src/models/builtin.cpp) expands, one entry per built model.
set(_halo_registrations "")
foreach(_m IN LISTS HALO_MODELS)
  string(APPEND _halo_registrations "HALO_BUILT_MODEL(${_m})\n")
endforeach()
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/builtin_models.inc CONTENT "${_halo_registrations}" @ONLY)

# The registry names the models in its error for a pack whose model this build left out.
string(REPLACE ";" "," _halo_known "${HALO_BUILTIN_MODELS}")
string(REPLACE ";" "," _halo_built "${HALO_MODELS}")
