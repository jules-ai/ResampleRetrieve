#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include "resample_retrieval.h"
#include "helper.h"

namespace fs = std::filesystem;

double get_cluster_threshold(const std::string& config_file, double default_val = 0.6) {
    std::ifstream infile(config_file);
    if (!infile.is_open()) return default_val;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.find("cluster_threshold") != std::string::npos) {
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                try {
                    return std::stod(line.substr(pos + 1));
                } catch(...) {
                    return default_val;
                }
            }
        }
    }
    return default_val;
}

int main() {
    double threshold = get_cluster_threshold("config.ini", 0.6);
    std::cout << "Cluster threshold: " << threshold << std::endl;

    std::string images_dir = "images";
    if (!fs::exists(images_dir)) {
        std::cerr << "Directory 'images' does not exist." << std::endl;
        return 1;
    }

    std::vector<std::string> images = jules::get_images(images_dir);
    if (images.empty()) {
        std::cerr << "No images found in 'images' directory." << std::endl;
        return 1;
    }

    std::string clusters_dir = "clusters";
    fs::create_directories(clusters_dir);

    jules::ResampleRetrieval retriever;
    if (0 != retriever.Init(jules::ModelType::MOBILENET_V4_S)) {
        std::cerr << "Failed to init model." << std::endl;
        return 1;
    }

    for (const auto& img : images) {
        retriever.Push(img);
    }

    std::unordered_map<std::string, bool> visited;
    for (const auto& img : images) {
        visited[img] = false;
    }

    int cluster_id = 0;
    for (const auto& img : images) {
        if (visited[img]) continue;

        std::vector<jules::Result> results;
        if (0 != retriever.Query(img, results, images.size())) {
            std::cerr << "Query failed for " << img << std::endl;
            continue;
        }

        std::vector<std::string> cluster_members;
        for (const auto& res : results) {
            if (res.similarity >= threshold && !visited[res.path]) {
                cluster_members.push_back(res.path);
                visited[res.path] = true;
            }
        }

        if (!cluster_members.empty()) {
            std::string current_cluster_dir = clusters_dir + "/cluster_" + std::to_string(cluster_id);
            fs::create_directories(current_cluster_dir);
            for (const auto& member : cluster_members) {
                fs::copy_file(member, current_cluster_dir + "/" + fs::path(member).filename().string(), fs::copy_options::overwrite_existing);
            }
            cluster_id++;
        }
    }

    std::cout << "Clustered " << images.size() << " images into " << cluster_id << " clusters." << std::endl;
    return 0;
}
