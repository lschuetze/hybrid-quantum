add_mlir_translation_library(QuantumCircuitMetrics
    QuantumCircuitMetricsRegistration.cpp
    QuantumCircuitMetrics.cpp

    DEPENDS
    QuantumIR

    LINK_LIBS PUBLIC
    MLIRIR
    QuantumIR
    )
