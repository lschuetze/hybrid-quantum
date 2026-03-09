add_mlir_conversion_library(ConvertRVSDG
        RVSDGConversion.cpp

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
