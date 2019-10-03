#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	void train(int n, int *odata, const int *idata);
	void train_cpu(int n, float *data, const float expected);
}
