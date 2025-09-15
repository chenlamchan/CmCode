#include "StdAfx.h"
#include "CmAdaptiveTripleThresh.h"

CmAdaptiveTripleThresh::CmAdaptiveTripleThresh(CMat& img3f, CMat& sal1f)
    : _img3f(img3f), _sal1f(sal1f), _w(img3f.cols), _h(img3f.rows)
{
    CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
    CV_Assert(sal1f.data != NULL && sal1f.type() == CV_32FC1);
    CV_Assert(img3f.size() == sal1f.size());

    // Set window size - use half the width as mentioned in the document
    _windowSize = max(2, _w / 2);
    //if (_windowSize % 2 == 1) _windowSize++; // Ensure even window size

    // Find global maximum saliency value L
    double minVal;
    minMaxLoc(_sal1f, &minVal, &_maxSaliency);

    // Build integral image for fast sum computation
    buildIntegralImage();
}

CmAdaptiveTripleThresh::~CmAdaptiveTripleThresh(void)
{
    // Nothing to cleanup
}

void CmAdaptiveTripleThresh::buildIntegralImage()
{
    // Build integral image for fast rectangle sum computation
    try {
        integral(_sal1f, _integralImg, CV_64F);
    }
    catch (const cv::Exception& e) {
        printf("Error building integral image: %s\n", e.what());
        throw;
    }
}

double CmAdaptiveTripleThresh::calculateRectangleSum(int x1, int y1, int x2, int y2)
{
    // Ensure coordinates are within bounds
    x1 = max(0, min(x1, _w - 1));
    y1 = max(0, min(y1, _h - 1));
    x2 = max(x1, min(x2, _w - 1));
    y2 = max(y1, min(y2, _h - 1));

    if (x2 + 1 >= _integralImg.cols || y2 + 1 >= _integralImg.rows) {
        printf("Warning: Integral image bounds exceeded\n");
        return 0.0;
    }

    // Calculate sum using integral image formula: I(x2,y2) - I(x1-1,y2) - I(x2,y1-1) + I(x1-1,y1-1)
    double sum = _integralImg.at<double>(y2 + 1, x2 + 1);

    if (x1 > 0) sum -= _integralImg.at<double>(y2 + 1, x1);
    if (y1 > 0) sum -= _integralImg.at<double>(y1, x2 + 1);
    if (x1 > 0 && y1 > 0) sum += _integralImg.at<double>(y1, x1);

    return max(0.0,sum);
}

Mat CmAdaptiveTripleThresh::performAdaptiveThresholding()
{
    Mat classificationMap = Mat::zeros(_h, _w, CV_8UC1);

    // Process each pixel according to the flowchart
    for (int y = 0; y < _h; y++) {
        const float* salRow = _sal1f.ptr<float>(y);
        uchar* classRow = classificationMap.ptr<uchar>(y);

        for (int x = 0; x < _w; x++) {
            // Define window boundaries centered on current pixel
            int S = _windowSize;
            int x1 = x - S / 2;
            int x2 = x + S / 2;
            int y1 = y - S / 2;
            int y2 = y + S / 2;

            //// Calculate sum and local mean ti
            double Rs = calculateRectangleSum(x1, y1, x2, y2);

            // Calculate actual window boundaries (clipped to image)
            int actualX1 = max(0, x1);
            int actualY1 = max(0, y1);
            int actualX2 = min(_w - 1, x2);
            int actualY2 = min(_h - 1, y2);

            // Calculate number of pixels in window
            int N = (actualX2 - actualX1 + 1) * (actualY2 - actualY1 + 1);

            double localMean = Rs / N; // This is ti

            // Get current pixel saliency value
            float pixelSaliency = salRow[x];

            // Follow the flowchart classification logic
            if (pixelSaliency <= localMean) {
                // Certain Background
                classRow[x] = CERTAIN_BACKGROUND;
            }
            else {
                // Calculate th (high threshold): th = ti + (L - ti)/2
                double th = localMean + ((_maxSaliency - localMean) / 2.0);

                if (pixelSaliency > th) {
                    // Certain Foreground  
                    classRow[x] = CERTAIN_FOREGROUND;
                }
                else {
                    // Calculate tm (medium threshold): tm = ti + (th - ti)/2
                    double tm = localMean + ((th - localMean) / 2.0);

                    if (pixelSaliency <= tm) {
                        // Probable Background
                        classRow[x] = PROBABLE_BACKGROUND;
                    }
                    else {
                        // Probable Foreground
                        classRow[x] = PROBABLE_FOREGROUND;
                    }
                }
            }
        }
    }

    return classificationMap;
}

Mat CmAdaptiveTripleThresh::generateGrabCutMask(CMat& classificationMap)
{
    Mat grabCutMask = Mat::zeros(_h, _w, CV_8UC1);

    for (int y = 0; y < _h; y++) {
        const uchar* classRow = classificationMap.ptr<uchar>(y);
        uchar* maskRow = grabCutMask.ptr<uchar>(y);

        for (int x = 0; x < _w; x++) {
            uchar classification = classRow[x];

            switch (classification) {
            case CERTAIN_BACKGROUND:
                maskRow[x] = cv::GC_BGD; // Definite background
                break;
            case PROBABLE_BACKGROUND:
                maskRow[x] = cv::GC_PR_BGD; // Probable background  
                break;
            case PROBABLE_FOREGROUND:
                maskRow[x] = cv::GC_PR_FGD; // Probable foreground
                break;
            case CERTAIN_FOREGROUND:
                maskRow[x] = cv::GC_FGD; // Definite foreground
                break;

            }
        }
    }

    return grabCutMask;
}

Mat CmAdaptiveTripleThresh::applyGrabCut(CMat& classificationMap, int iterations)
{
    Mat grabCutMask = generateGrabCutMask(classificationMap);
    Mat bgdModel, fgdModel;

    // Convert image to 8-bit for GrabCut
    Mat img8u;
    _img3f.convertTo(img8u, CV_8UC3, 255);

    // Apply GrabCut algorithm
    try {
        cv::grabCut(img8u, grabCutMask, Rect(), bgdModel, fgdModel, iterations, cv::GC_INIT_WITH_MASK);
    }
    catch (const cv::Exception& e) {
        printf("GrabCut failed: %s\n", e.what());
        return Mat();
    }

    // Generate final binary mask (foreground pixels = 255, background = 0)
    Mat finalMask;
    //compare(grabCutMask & 1, 0, finalMask, CMP_NE);

    finalMask = (grabCutMask == cv::GC_FGD) | (grabCutMask == cv::GC_PR_FGD);

    return finalMask;
}

Mat CmAdaptiveTripleThresh::processImage(int grabCutIterations)
{
    // Step 1: Perform adaptive triple thresholding
    Mat classificationMap = performAdaptiveThresholding();

    // Step 2: Apply GrabCut with adaptive initialization
    Mat finalMask = applyGrabCut(classificationMap, grabCutIterations);

    return finalMask;
}

Mat CmAdaptiveTripleThresh::visualizeClassification(CMat& classificationMap)
{
    Mat colorMap = Mat::zeros(_h, _w, CV_8UC3);

    for (int y = 0; y < _h; y++) {
        const uchar* classRow = classificationMap.ptr<uchar>(y);
        Vec3b* colorRow = colorMap.ptr<Vec3b>(y);

        for (int x = 0; x < _w; x++) {
            uchar classification = classRow[x];

            switch (classification) {
            case CERTAIN_BACKGROUND:
                colorRow[x] = Vec3b(0, 0, 0); // Black
                break;
            case PROBABLE_BACKGROUND:
                colorRow[x] = Vec3b(128, 0, 0); // Dark blue
                break;
            case PROBABLE_FOREGROUND:
                colorRow[x] = Vec3b(0, 128, 255); // Orange
                break;
            case CERTAIN_FOREGROUND:
                colorRow[x] = Vec3b(0, 255, 255); // Yellow
                break;
            }
        }
    }

    return colorMap;
}

void CmAdaptiveTripleThresh::printStatistics(CMat& classificationMap)
{
    vector<int> counts(4, 0);
    int totalPixels = _w * _h;

    for (int y = 0; y < _h; y++) {
        const uchar* classRow = classificationMap.ptr<uchar>(y);
        for (int x = 0; x < _w; x++) {
            counts[classRow[x]]++;
        }
    }

    printf("Classification Statistics:\n");
    printf("Certain Background: %d (%.1f%%)\n", counts[0], 100.0 * counts[0] / totalPixels);
    printf("Probable Background: %d (%.1f%%)\n", counts[1], 100.0 * counts[1] / totalPixels);
    printf("Probable Foreground: %d (%.1f%%)\n", counts[2], 100.0 * counts[2] / totalPixels);
    printf("Certain Foreground: %d (%.1f%%)\n", counts[3], 100.0 * counts[3] / totalPixels);
}

Mat CmAdaptiveTripleThresh::CutObjsAdaptive(CMat& img3f, CMat& sal1f, int windowSizeFactor, int grabCutIterations)
{
    CV_Assert(img3f.data != NULL && sal1f.data != NULL);
    CV_Assert(img3f.size() == sal1f.size());

    // Ensure saliency map is normalized to [0,1]
    Mat normalizedSal;
    double minVal, maxVal;
    minMaxLoc(sal1f, &minVal, &maxVal);
    if (maxVal > 1.1) {
        sal1f.convertTo(normalizedSal, CV_32F, 1.0 / 255.0);
    }
    else {
        normalizedSal = sal1f;
    }

    // Create adaptive triple thresholding processor
    CmAdaptiveTripleThresh processor(img3f, normalizedSal);

    // Process and return binary mask
    Mat result = processor.processImage(grabCutIterations);

    return result;
}

