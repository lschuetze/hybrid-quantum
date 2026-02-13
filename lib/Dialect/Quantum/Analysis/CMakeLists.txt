add_mlir_library(QuantumAnalysis
    RegisterRangesAnalysis.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QuantumIncGen
)
