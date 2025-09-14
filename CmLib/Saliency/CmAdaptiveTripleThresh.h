#pragma once

/************************************************************************/
/* Adaptive Triple Thresholding for Salient Object Segmentation        */
/* Based on integral image computation for efficient local mean calc    */
/************************************************************************/

class CmAdaptiveTripleThresh
{
public:
    // Classification types for pixels
    enum RegionType {
        CERTAIN_BACKGROUND = 0,
        PROBABLE_BACKGROUND = 1,
        PROBABLE_FOREGROUND = 2,
        CERTAIN_FOREGROUND = 3
    };

    // GrabCut mask values
    enum GrabCutMask {
        GC_BGD = 0,     // Definite background
        GC_FGD = 1,      // Definite foreground
        GC_PR_BGD = 2,  // Probable background
        GC_PR_FGD = 3,  // Probable foreground
    };

public:
    CmAdaptiveTripleThresh(CMat& img3f, CMat& sal1f);
    ~CmAdaptiveTripleThresh(void);

    // Main processing function - returns binary segmentation mask
    Mat processImage(int grabCutIterations = 5);

    // Individual steps of the algorithm
    Mat performAdaptiveThresholding();
    Mat generateGrabCutMask(CMat& classificationMap);
    Mat applyGrabCut(CMat& classificationMap, int iterations = 5);

    // Utility functions
    Mat visualizeClassification(CMat& classificationMap);
    void printStatistics(CMat& classificationMap);

    // Static method for easy integration with existing code
    static Mat CutObjsAdaptive(CMat& img3f, CMat& sal1f, int windowSizeFactor = 2,
        int grabCutIterations = 5);

private:
    void buildIntegralImage();
    double calculateRectangleSum(int x1, int y1, int x2, int y2);

private:
    Mat _img3f;          // Original BGR image
    Mat _sal1f;          // Saliency map
    Mat _integralImg;    // Integral image for fast computation
    int _w, _h;          // Image dimensions
    int _windowSize;     // Adaptive window size
    double _maxSaliency; // Global maximum saliency value L
};