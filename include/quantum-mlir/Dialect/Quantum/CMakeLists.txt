################################################################################
# QuantumIncGen
#
# MLIR Quantum dialect generated includes.
################################################################################

add_custom_target(QuantumIncGen)

mlir_gen_enums(Quantum)

add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)
