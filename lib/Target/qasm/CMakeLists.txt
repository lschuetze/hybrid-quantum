add_mlir_translation_library(QILLRToOpenQASM
    TargetQASMRegistration.cpp
    TargetQASM.cpp

    DEPENDS
    QILLRIR

    LINK_LIBS PUBLIC
    MLIRIR
    QILLRIR
    )

#target_link_libraries(QIRToOpenQASM PUBLIC MLIRIR)
