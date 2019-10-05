set(SOURCE_FILES
    "aes_cbr.h"
    "aes_cbr.cu"
    )

cuda_add_library(aes
    ${SOURCE_FILES}
    OPTIONS -arch=sm_70
    )
