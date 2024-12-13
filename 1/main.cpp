#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

void show(cv::Mat& image, const std::string& title = "Image")
{
    namedWindow(title, cv::WINDOW_AUTOSIZE);

    imshow(title, image);
}

cv::Mat resize_image(cv::Mat& image, int coeff)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image.cols / coeff, image.rows / coeff));

    return resized_image;
}

cv::Mat remove_shadows(cv::Mat& image, int kernel_scale_factor)
{
    const int gaussian_kernel_size = image.size().width / kernel_scale_factor | 1;

    cv::Mat lightmap;
    cv::GaussianBlur(image, lightmap, cv::Size(gaussian_kernel_size, gaussian_kernel_size), 0.0);

    cv::Mat result;
    cv::divide(image, lightmap, result, image.size().width);

    return result;
}

cv::Mat find_edges(cv::Mat& image)
{
    cv::Mat result;
    cv::Canny(image, result, 150, 150);

    return result;
}

cv::Mat distances(cv::Mat& image)
{
    cv::Mat result;
    cv::Mat labels;
    cv::distanceTransform(image, result, labels, cv::DistanceTypes::DIST_C, 3, 5);

    return labels;
}

cv::Mat match(cv::Mat& image, const cv::Mat& template_image)
{
    cv::Mat result;
    cv::matchTemplate(image, template_image, result, cv::TemplateMatchModes::TM_CCOEFF);
    cv::normalize(result, result, 1, 0, cv::NormTypes::NORM_MINMAX);

    return result;
}

cv::Mat remove_color_at_bounds(cv::Mat& image, int x, int y, int bounds)
{
    cv::Mat hsv_image; cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    auto color = hsv_image.at<cv::Vec3b>(x, y);

    auto bound = [](uchar v) { return v <= 255 ? v : 255; };

    cv::Vec3b lower = cv::Vec3b(bound(color[0] - bounds), 0, 0);
    cv::Vec3b upper = cv::Vec3b(bound(color[0] + bounds), 255, 255);

    cv::Mat mask; cv::inRange(hsv_image, lower, upper, mask);
    mask = 255 - mask;

    cv::Mat removed; cv::bitwise_and(hsv_image, hsv_image, removed, mask);

    return removed;
}

cv::Mat morph(cv::Mat& image, int kernel_size)
{
    cv::Mat kernel = cv::Mat::ones(cv::Size(kernel_size, kernel_size), CV_8U);
    cv::Mat result; 
    cv::morphologyEx(image, result, cv::MorphTypes::MORPH_DILATE, kernel);

    return result;
}

double calc_elongation(cv::Moments& m)
{
    double sqrt = std::sqrt((m.m20 - m.m02) * (m.m20 - m.m02) + 4 * m.m11 * m.m11);
    double d1 = m.m20 + m.m02 + sqrt;
    double d2 = m.m20 + m.m02 - sqrt;
    return d1 / d2;
}

void parse_contours(cv::Mat& image, cv::Mat& result)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contours_image(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    struct contours_data {
        std::vector<cv::Point_<int>> c;
        double solidity, elongation;
        int x, y, width, height;

        contours_data(std::vector<cv::Point_<int>> c, double s, double e, int x, int y, int w, int h)
            : c(std::move(c)), solidity(s), elongation(e), x(x), y(y), width(w), height(h) {}
    };
    std::vector<contours_data> cd;
    std::vector<double> elongations;

    for (auto& c : contours)
    {
        double area = cv::contourArea(c, false);
        cv::Mat hull; cv::convexHull(c, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = area / hull_area;

        auto m = cv::moments(c, false);

        double elongation = calc_elongation(m);

        cv::Rect r = cv::boundingRect(c);

        cd.push_back({c, solidity, 0.0, r.x, r.y, r.width, r.height});
        elongations.push_back(elongation);
    }

    std::vector<double> normalized_elongations;
    cv::normalize(elongations, normalized_elongations);

    for (int i = 0; i < cd.size(); ++i) {
        cd[i].elongation = normalized_elongations[i];
    }

    for (auto &[c, s, e, x, y, w, h] : cd) {
        std::cout << y << " " << x << " " << s << " " << e << " " << std::endl;
        std::vector<std::vector<cv::Point>> contours(1, c);

        cv::Vec3b color;
        if (s > 0.94 && e < 0.4) {
            color = cv::Vec3b(255, 0, 0);
        } else {
            color = cv::Vec3b(0, 255, 0);
        }
        cv::drawContours(result, contours, -1, color, 2);

//        std::string s_text;
//        s_text += "S: ";
//        s_text += std::to_string(s).substr(0, 5);
//        std::string e_text;
//        e_text += "  E: ";
//        e_text += std::to_string(e).substr(0, 5);
//        cv::putText(result, s_text, cv::Point(x, y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 0), 2);
//        cv::putText(result, e_text, cv::Point(x, y + 20), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 0), 2);
    }

    show(result);
}

void parse_components(cv::Mat& image)
{
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(image, labels, stats, centroids);

    cv::Mat areas(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::RNG rng(time(0));
    int components_count = stats.rows;
    cv::Mat colors(1, components_count, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int x = 1; x < components_count; ++x)
    {
        colors.at<cv::Vec3b>(0, x) = cv::Vec3b(
            rng.uniform(0, 255),
            rng.uniform(0, 255),
            rng.uniform(0, 255)
        );
    }

    int max_area = 0;
    std::vector<double> elongations(components_count + 1);

    for (int c = 1; c < components_count; ++c)
    {
        int x = stats.at<int>(cv::Point(0, c));
        int y = stats.at<int>(cv::Point(1, c));
        int width = stats.at<int>(cv::Point(2, c));
        int height = stats.at<int>(cv::Point(3, c));
        int area = stats.at<int>(cv::Point(4, c));

        max_area = std::max(max_area, area);

        int roi_width = x + width > image.cols ? x + width - image.cols : x + width;
        int roi_height = y + height > image.rows ? y + height - image.rows : y + height;

        std::cout << x << " " << y << " " << width << " " << height << " " << roi_width << " " << roi_height << std::endl;

        cv::Mat region = image(cv::Range(y, roi_height), cv::Range(x, roi_width));

        cv::Mat corners; cv::goodFeaturesToTrack(region, corners, 100, 0.001, 30);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(region, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        cv::Mat polys; cv::approxPolyDP(contours[0], polys, 0.1 * cv::arcLength(contours[0], true), true);

       /* std::vector<std::vector<cv::Point>> contours;
        cv::findContours(region, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        cv::Mat polys; cv::approxPolyDP(contours[0], polys, 0.01 * cv::arcLength(contours[0], true), true);

        if (polys.rows < 10)
        {
            cv::Rect r = cv::boundingRect(contours[0]);
            double ratio = (double)(r.width) / r.height;
            elongations[c] = 1;
            continue;
        }
        else
        {
            elongations[c] = 0;
            continue;
        }

        auto m = cv::moments(contours[0], true);
        double elongation = calc_elongation(m);*/

        //cv::drawContours(image, contours, 0, cv::Vec3b(100, 100, 100), 1, 8, cv::noArray(), 2147483647, cv::Point(x, y));

        //std::cout << elongation * cv::arcLength(contours[0], true) << std::endl;

        std::cout << corners.rows << std::endl;
        //elongations[c] = elongation * cv::arcLength(contours[0], true);

        elongations[c] = polys.rows;

      /*  show(image, "image");
        cv::waitKey(0);*/
    }

    //cv::normalize(elongations, elongations, 1, cv::NormTypes::NORM_MINMAX);

    for (int x = 0; x < labels.rows; ++x)
    {
        for (int y = 0; y < labels.cols; ++y)
        {
            int component_id = labels.at<int>(x, y);
            if (component_id == 0) continue;

            double elongation = elongations[component_id];

            cv::Vec3b& color = areas.at<cv::Vec3b>(x, y);

            int width = stats.at<int>(cv::Point(cv::CC_STAT_WIDTH, component_id));
            int height = stats.at<int>(cv::Point(cv::CC_STAT_HEIGHT, component_id));
            int area = stats.at<int>(cv::Point(cv::CC_STAT_AREA, component_id));

            double coeff = 0.7;

            if (elongation > 0.2)
            {
                color = colors.at<cv::Vec3b>(0, 2);
            }
            else
            {
                color = colors.at<cv::Vec3b>(0, 3);

            }
        }
    }

    show(areas, "image");
}

void task1()
{
    cv::Mat image = cv::imread("samples/1.jpg", cv::IMREAD_COLOR);
    cv::Mat template_image = cv::imread("samples/1-3.template.jpg", cv::IMREAD_COLOR);

    cv::Mat resized_image = resize_image(image, 2);
    cv::Mat resized_template = resize_image(template_image, 2);

    cv::Mat gs_image; cv::cvtColor(resized_image, gs_image, cv::IMREAD_GRAYSCALE);
    cv::Mat gs_template_image; cv::cvtColor(resized_template, gs_template_image, cv::IMREAD_GRAYSCALE);

    cv::Mat image_edges = find_edges(gs_image);
    cv::Mat template_edges = find_edges(gs_template_image);

    cv::Mat correlation_map = match(image_edges, template_edges);

    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(correlation_map, &min, &max, &min_loc, &max_loc);

    cv::Mat image_with_match; resized_image.copyTo(image_with_match);
    cv::Rect match_rect(max_loc.x, max_loc.y, resized_template.cols, resized_template.rows);
    cv::rectangle(image_with_match, match_rect, 255, 2);

    namedWindow("Image", cv::WINDOW_AUTOSIZE);

    cv::Mat results;
    cv::hconcat(resized_image, image_with_match, results);

    imshow("Image", correlation_map);

    cv::imwrite("samples/1.image.edges.jpg", image_edges);
    cv::imwrite("samples/1.template.edges.jpg", template_edges);
    cv::imwrite("samples/1.correlation.tiff", correlation_map);
    cv::imwrite("samples/1.results.jpg", results);
}

void task2()
{
    cv::Mat image = cv::imread("samples/2_1.jpg", cv::IMREAD_COLOR);

    cv::Mat resized_image = resize_image(image, 2);

    show(resized_image);
    cv::waitKey();

    cv::Mat bg_removed = remove_color_at_bounds(resized_image, 0, 0, 5);

    show(bg_removed);
    cv::waitKey(0);
    cv::destroyAllWindows();


    cv::Mat gs_bg_removed; cv::cvtColor(bg_removed, gs_bg_removed, cv::COLOR_BGR2GRAY);

    show(gs_bg_removed);
    cv::waitKey(0);


    cv::Mat binary_raw; cv::threshold(gs_bg_removed, binary_raw, 0, 255, cv::ThresholdTypes::THRESH_TRIANGLE);

    show(binary_raw);
    cv::waitKey(0);

    cv::Mat binary = morph(binary_raw, 6);

    show(binary);
    cv::waitKey(0);

    parse_contours(binary, resized_image);
    //parse_components(binary);
}

int main()
{
    //task1();
    task2();

    cv::waitKey(0);
    return 0;
}