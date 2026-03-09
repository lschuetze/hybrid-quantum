add_mlir_library(QuantumInterfaces
    InferRegisterRangesInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QuantumIncGen

    LINK_LIBS PUBLIC
        QuantumAnalysis
)
