add_mlir_library(QuantumAnalysis
    RegisterRangesAnalysis.cpp
    PhasePolynomialAnalysis.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QuantumIncGen
)
