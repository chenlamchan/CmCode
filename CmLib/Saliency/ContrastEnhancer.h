#ifndef CONTRAST_ENHANCER_H
#define CONTRAST_ENHANCER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

/**
 * @brief Applies Global Contrast Enhancement using Histogram Equalization on the luminance channel
 * As per Yoon et al. (2018)'s paper.
 */
class ContrastEnhancer {
public:
    /**
     * @brief Constructor
     */
    ContrastEnhancer();

    /**
     * @brief Enhance the contrast of a BGR image
     * @param imageBGR Input BGR image
     * @return Enhanced image in the same format as input, or empty Mat on error
     */
    Mat process(const Mat& imageBGR);

    /**
     * @brief Demo function to process multiple images from a directory
     * @param imgDir Input directory containing images (e.g., "C:/Images")
     * @param outputDir Output directory for enhanced images
     * @return Number of successfully processed images, -1 on error
     */
    static int Demo(CStr imgDir, CStr outputDir);

    /**
     * @brief Demo function to process a single image
     * @param inputImagePath Path to input image
     * @param outputDir Output directory for results
     * @return 1 on success, -1 on error
     */
    static int DemoSingle(CStr inputImagePath, CStr outputDir);

private:
    /**
     * @brief Converts input image to uint8 format if needed
     * @param image Input image
     * @return Converted image in uint8 format
     */
    Mat convertToUint8(const Mat& image);

    /**
     * @brief Converts image back from uint8 to original type
     * @param image Input image in uint8 format
     * @param originalType Original image type
     * @return Converted image in original type
     */
    Mat convertFromUint8(const Mat& image, int originalType);

    /**
     * @brief Validates input image format
     * @param image Input image to validate
     * @return true if valid, false otherwise
     */
    bool isValidInput(const Mat& image);
};

#endif