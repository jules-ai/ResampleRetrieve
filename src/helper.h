#ifndef JULES_HELPER_H
#define JULES_HELPER_H

#include "resample_retrieval.h"

namespace jules
{
    std::vector<std::string> get_images(const std::string &directory_path);

    cv::Mat visualize(const std::vector<std::vector<jules::Result>> &all_results,
                      const std::vector<std::string> &target_files,
                      const int topk);
}

#endif // JULES_HELPER_H