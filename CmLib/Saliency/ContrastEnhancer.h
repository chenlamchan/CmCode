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
     * @brief Enhance the contrast of a BGR image using custom implementation
     * @param imageBGR Input BGR image
     * @return Enhanced image in the same format as input, or empty Mat on error
     */
    Mat process(const Mat& imageBGR);

    /**
     * @brief Enhance the contrast of a BGR image using OpenCV implementation
     * @param imageBGR Input BGR image
     * @return Enhanced image in the same format as input, or empty Mat on error
     */
    Mat processOptimized(const Mat& imageBGR);

    /**
     * @brief Enhance the contrast of a BGR image using OpenCV implementation, but with custom global hist eq
     * @param imageBGR Input BGR image
     * @return Enhanced image in the same format as input, or empty Mat on error
     */
    Mat processOptimizedcustomeq(const Mat& imageBGR);

    /**
     * @brief Combined local and global contrast enhancement as per the paper
     * Step 1: Apply local contrast stretching
     * Step 2: Apply global contrast stretching to the local result
     * Step 3: Combine both methods for final enhancement
     * @param imageBGR Input BGR image
     * @return Enhanced image using combined local+global method
     */
    Mat processLocalGlobalCombined(const Mat& imageBGR);

   
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

    /**
     * @brief Custom BGR to YCrCb conversion following the paper's exact equations
     * Uses the specific coefficients and delta values mentioned in the research paper:
     * Y = 0.299*R + 0.587*G + 0.114*B
     * Cr = (R-Y)*0.713 + 128
     * Cb = (B-Y)*0.564 + 128
     * @param bgr Input BGR image (CV_8UC3)
     * @return YCrCb image with paper-specific conversion
     */
    Mat customBGRtoYCrCb(const Mat& bgr);

    /**
     * @brief Custom YCrCb to BGR conversion (inverse of customBGRtoYCrCb)
     * Applies the inverse transformation to convert back to BGR color space
     * @param ycrcb Input YCrCb image (CV_8UC3)
     * @return BGR image
     */
    Mat customYCrCbtoBGR(const Mat& ycrcb);

    /**
     * @brief Global contrast enhancement using contrast stretching
     * Replaces histogram equalization with linear contrast stretching
     * @param src Input single-channel image (CV_8UC1)
     * @param dst Output enhanced image
     */
    void globalContrastEnhancement(const Mat& src, Mat& dst);

    /**
     * @brief Local contrast stretching for enhancing local features
     * Uses unsharp masking technique as described in the paper
     * @param src Input single-channel image
     * @param dst Output locally enhanced image
     * @param kernelSize Size of the Gaussian kernel for unsharp masking
     * @param sigma Gaussian blur sigma
     * @param amount Unsharp mask strength
     * @param threshold Unsharp mask threshold
     */
    void localContrastStretching(const Mat& src, Mat& dst,
        int kernelSize = 15,
        double sigma = 5.0,
        double amount = 1.5,
        double threshold = 0);

    
};

#endif