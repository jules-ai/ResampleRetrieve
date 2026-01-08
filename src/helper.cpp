#include "helper.h"
#include <filesystem>
namespace fs = std::filesystem;
std::vector<std::string> jules::get_images(const std::string &directory_path)
{
    auto is_image = [](const fs::path &path)
    {
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (fs::is_regular_file(path) &&
            (ext == ".jpg" ||
             ext == ".jpeg" ||
             ext == ".png"))
            return true;

        return false;
    };
    std::vector<std::string> images;
    if (fs::is_directory(directory_path))
    {
        for (const auto &entry : fs::directory_iterator(directory_path))
        {
            if (is_image(entry))
            {
                images.push_back(entry.path().string());
            }
        }
    }
    return images;
}

cv::Mat jules::visualize(const std::vector<std::vector<jules::Result>> &all_results, const std::vector<std::string> &target_files, const int topk)
{
    constexpr int cell_width = 200;
    constexpr int cell_height = 200;
    constexpr int header_height = 40;
    constexpr int border_thickness = 2;

    cv::Mat table_image = cv::Mat::zeros(
        static_cast<int>(target_files.size()) * cell_height + static_cast<int>(target_files.size() + 2) * border_thickness + header_height,
        (topk + 1) * cell_width + (topk + 2) * border_thickness,
        CV_8UC3);
    table_image = cv::Scalar(255, 255, 255);

    std::vector<std::string> headers = {"Query", "Top-1", "Top-2", "Top-3"};
    for (int col = 0; col < headers.size(); ++col)
    {
        cv::Rect header_rect(
            col * (cell_width + border_thickness) + border_thickness,
            0,
            cell_width,
            header_height);

        cv::rectangle(table_image, header_rect, cv::Scalar(200, 200, 200), -1);

        cv::putText(
            table_image,
            headers[col],
            cv::Point(header_rect.x + 10, header_rect.y + header_rect.height / 2 + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            1);

        cv::rectangle(table_image, header_rect, cv::Scalar(0, 0, 0), border_thickness);
    }

    for (int row = 0; row < target_files.size(); ++row)
    {
        cv::Mat query_img = cv::imread(target_files[row]);
        if (!query_img.empty())
        {
            cv::resize(query_img, query_img, cv::Size(cell_width - 20, cell_height - 20));

            cv::Rect query_rect(
                border_thickness,
                row * (cell_height + border_thickness) + header_height,
                query_img.cols,
                query_img.rows);

            if (query_rect.y + query_rect.height < table_image.rows &&
                query_rect.x + query_rect.width < table_image.cols)
            {
                cv::Mat roi = table_image(cv::Rect(query_rect.x, query_rect.y, query_rect.width, query_rect.height));
                query_img.copyTo(roi);
            }
        }

        cv::Rect query_cell_rect(
            border_thickness,
            row * (cell_height + border_thickness) + header_height,
            cell_width,
            cell_height);
        cv::rectangle(table_image, query_cell_rect, cv::Scalar(0, 0, 0), border_thickness);

        if (row < all_results.size())
        {
            for (int col = 0; col < all_results[row].size(); ++col)
            {
                cv::Mat result_img = cv::imread(all_results[row][col].path);
                if (!result_img.empty())
                {
                    cv::resize(result_img, result_img, cv::Size(cell_width - 20, cell_height - 20));

                    int x_pos = (col + 1) * (cell_width + border_thickness) + border_thickness + 10;
                    int y_pos = row * (cell_height + border_thickness) + header_height + 10;

                    if (y_pos + result_img.rows < table_image.rows &&
                        x_pos + result_img.cols < table_image.cols)
                    {
                        cv::Mat roi = table_image(cv::Rect(x_pos, y_pos, result_img.cols, result_img.rows));
                        result_img.copyTo(roi);
                    }
                }

                cv::Rect result_cell_rect(
                    (col + 1) * (cell_width + border_thickness) + border_thickness,
                    row * (cell_height + border_thickness) + header_height,
                    cell_width,
                    cell_height);
                cv::rectangle(table_image, result_cell_rect, cv::Scalar(0, 0, 0), border_thickness);

                std::ostringstream score_str;
                score_str << std::fixed << std::setprecision(3) << all_results[row][col].similarity;
                std::string label = score_str.str();
                cv::rectangle(table_image, cv::Rect(result_cell_rect.x + 5, result_cell_rect.y + cell_height - 15, 32, 10), cv::Scalar(255, 255, 255), cv::FILLED);

                cv::putText(
                    table_image,
                    label,
                    cv::Point(result_cell_rect.x + 5, result_cell_rect.y + cell_height - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cv::Scalar(0, 0, 0),
                    1);
            }
        }
    }

    for (int row = 0; row <= target_files.size(); ++row)
    {
        int y_pos = row == 0 ? header_height : row * (cell_height + border_thickness) + header_height;
        cv::line(
            table_image,
            cv::Point(0, y_pos),
            cv::Point(table_image.cols, y_pos),
            cv::Scalar(0, 0, 0),
            border_thickness);
    }

    for (int col = 0; col <= topk + 1; ++col)
    {
        int x_pos = col * (cell_width + border_thickness);
        cv::line(
            table_image,
            cv::Point(x_pos, 0),
            cv::Point(x_pos, table_image.rows),
            cv::Scalar(0, 0, 0),
            border_thickness);
    }
    return table_image;
}

