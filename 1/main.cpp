#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;

cv::Mat resize_image(cv::Mat& image, int coeff)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image.cols / coeff, image.rows / coeff));

    return resized_image;
}

cv::Mat remove_shadows(cv::Mat& image)
{
    const int kernel_scale_factor = 3;
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
    cv::Canny(image, result, 250, 250);

    return result;
}

cv::Mat distances(cv::Mat& image)
{
    cv::Mat result;
    cv::Mat labels;
    cv::distanceTransform(image, result, labels, DistanceTypes::DIST_C, 3, 5);

    return labels;
}

cv::Mat match(cv::Mat& image, const cv::Mat& template_image)
{
    cv::Mat result;
    cv::matchTemplate(image, template_image, result, TemplateMatchModes::TM_CCOEFF);
    cv::normalize(result, result, 1, 0, NormTypes::NORM_MINMAX);

    return result;
}

int main()
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

    namedWindow("Image", WINDOW_AUTOSIZE);

    cv::Mat results;
    cv::hconcat(resized_image, image_with_match, results);

    imshow("Image", correlation_map);

    cv::imwrite("samples/1.image.edges.jpg", image_edges);
    cv::imwrite("samples/1.template.edges.jpg", template_edges);
    cv::imwrite("samples/1.correlation.tiff", correlation_map);
    cv::imwrite("samples/1.results.jpg", results);

    waitKey(0);
    return 0;
}