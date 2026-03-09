add_mlir_conversion_library(RVSDGToScf
        RVSDGToScf.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRIR
        RVSDGIR
)
