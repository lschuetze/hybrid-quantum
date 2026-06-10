add_mlir_library(QuantumInterfaces
    InferRegisterRangesInterface.cpp
    InferPhasePolynomialInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QuantumIncGen

    LINK_LIBS PUBLIC
        QuantumAnalysis
)
