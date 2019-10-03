set(SOURCE_FILES
    "common.h"
    "common.cu"
    )

cuda_add_library(aes
    ${SOURCE_FILES}
    OPTIONS -arch=sm_70
    )
