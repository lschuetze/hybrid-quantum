add_mlir_conversion_library(QuantumToQILLR
        QuantumToQILLR.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransforms
        MLIRTransformUtils
        QuantumIR
        QILLRIR
        MLIRIR
)
