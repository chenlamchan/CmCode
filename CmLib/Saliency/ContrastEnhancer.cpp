#include "StdAfx.h"
#include "ContrastEnhancer.h"

ContrastEnhancer::ContrastEnhancer() {
    // Constructor - no initialization needed
}

Mat ContrastEnhancer::convertToUint8(const Mat& image) {
    Mat result;

    if (image.type() == CV_8UC3) {
        return image.clone();
    }

    // Check if image is normalized (0-1 range)
    double minVal, maxVal;
    minMaxLoc(image, &minVal, &maxVal);

    if (maxVal <= 1.0 && minVal >= 0.0) {
        // Normalized image, scale to 0-255
        image.convertTo(result, CV_8UC3, 255.0);
    }
    else {
        // Convert to uint8
        image.convertTo(result, CV_8UC3);
    }

    return result;
}

Mat ContrastEnhancer::convertFromUint8(const Mat& image, int originalType) {
    Mat result;

    if (originalType == CV_8UC3) {
        return image.clone();
    }
    else if (originalType == CV_32FC3) {
        // Convert back to 32-bit float normalized to [0,1]
        image.convertTo(result, CV_32FC3, 1.0 / 255.0);
    }
    else {
        // Default conversion
        image.convertTo(result, originalType);
    }

    return result;
}

bool ContrastEnhancer::isValidInput(const Mat& image) {
    if (image.empty()) {
        cout << "Error: Invalid input image for Contrast Enhancer - image is empty." << endl;
        return false;
    }

    if (image.dims != 2) {
        cout << "Error: Invalid input image for Contrast Enhancer - not a 2D image." << endl;
        return false;
    }

    if (image.channels() != 3) {
        cout << "Error: Invalid input image for Contrast Enhancer - not a 3-channel image." << endl;
        return false;
    }

    return true;
}

Mat ContrastEnhancer::customBGRtoYCrCb(const Mat& bgr) {
    Mat ycrcb;
    bgr.copyTo(ycrcb);

    if (bgr.type() == CV_8UC3) {
        // For 8-bit images, use integer arithmetic for better performance
        for (int i = 0; i < bgr.rows; i++) {
            for (int j = 0; j < bgr.cols; j++) {
                Vec3b pixel = bgr.at<Vec3b>(i, j);
                int B = pixel[0];
                int G = pixel[1];
                int R = pixel[2];

                // Use integer arithmetic (multiply by 1000 for precision)
                int Y = (299 * R + 587 * G + 114 * B) / 1000;
                int Cr = ((R - Y) * 713) / 1000 + 128;
                int Cb = ((B - Y) * 564) / 1000 + 128;

                // Y ranges from 16 to 235, Cr and Cb from 16 to 240
                Y = max(16, min(235, Y));
                Cr = max(16, min(240, Cr));
                Cb = max(16, min(240, Cb));

                ycrcb.at<Vec3b>(i, j) = Vec3b(
                    saturate_cast<uchar>(Y),
                    saturate_cast<uchar>(Cr),
                    saturate_cast<uchar>(Cb)
                );
            }
        }
    }
    else if (bgr.type() == CV_32FC3) {

        // For float images, use floating-point arithmetic
        for (int i = 0; i < bgr.rows; i++) {
            for (int j = 0; j < bgr.cols; j++) {
                Vec3f pixel = bgr.at<Vec3f>(i, j);
                float B = pixel[0];
                float G = pixel[1];
                float R = pixel[2];

                float Y = 0.299f * R + 0.587f * G + 0.114f * B;
                float Cr = (R - Y) * 0.713f + 0.5f; // 0.5f for normalized range
                float Cb = (B - Y) * 0.564f + 0.5f;

                // Clamp to [0,1] for normalized images
                Y = max(0.0f, min(1.0f, Y));
                Cr = max(0.0f, min(1.0f, Cr));
                Cb = max(0.0f, min(1.0f, Cb));

                ycrcb.at<Vec3f>(i, j) = Vec3f(Y, Cr, Cb);
            }
        }
    }

    
    return ycrcb;
}

Mat ContrastEnhancer::customYCrCbtoBGR(const Mat& ycrcb) {
    Mat bgr;
    ycrcb.copyTo(bgr);

    if (ycrcb.type() == CV_8UC3) {
        for (int i = 0; i < ycrcb.rows; i++) {
            for (int j = 0; j < ycrcb.cols; j++) {
                Vec3b pixel = ycrcb.at<Vec3b>(i, j);
                int Y = pixel[0];
                int Cr = pixel[1];
                int Cb = pixel[2];

                // Inverse transformation using integer arithmetic
                int R = Y + ((Cr - 128) * 1000) / 713;
                int G = Y - ((114 * (Cb - 128)) / (587 * 564) * 1000) - ((299 * (Cr - 128)) / (587 * 713) * 1000);
                int B = Y + ((Cb - 128) * 1000) / 564;

                // Clamp to 0-255
                R = max(0, min(255, R));
                G = max(0, min(255, G));
                B = max(0, min(255, B));

                bgr.at<Vec3b>(i, j) = Vec3b(
                    saturate_cast<uchar>(B),
                    saturate_cast<uchar>(G),
                    saturate_cast<uchar>(R)
                );
            }
        }
    }
    else if (ycrcb.type() == CV_32FC3) {
        for (int i = 0; i < ycrcb.rows; i++) {
            for (int j = 0; j < ycrcb.cols; j++) {
                Vec3f pixel = ycrcb.at<Vec3f>(i, j);
                float Y = pixel[0];
                float Cr = pixel[1];
                float Cb = pixel[2];

                // Inverse transformation
                float R = Y + (Cr - 0.5f) / 0.713f;
                float G = Y - (0.114f * (Cb - 0.5f)) / (0.587f * 0.564f) - (0.299f * (Cr - 0.5f)) / (0.587f * 0.713f);
                float B = Y + (Cb - 0.5f) / 0.564f;

                // Clamp to [0,1]
                R = max(0.0f, min(1.0f, R));
                G = max(0.0f, min(1.0f, G));
                B = max(0.0f, min(1.0f, B));

                bgr.at<Vec3f>(i, j) = Vec3f(B, G, R);
            }
        }
    }

    
    return bgr;
}

Mat ContrastEnhancer::process(const Mat& imageBGR) {
    // Validate input
    if (!isValidInput(imageBGR)) {
        return Mat();
    }

    // Store original type to convert back later
    int originalType = imageBGR.type();

    // Convert to YCrCb in original precision
    Mat ycrcb;
    try {
        ycrcb = customBGRtoYCrCb(imageBGR);
    }
    catch (const Exception& e) {
        cout << "Error converting BGR to YCrCb: " << e.what() << endl;
        return Mat();
    }

    if (ycrcb.empty()) {
        return Mat();
    }

    // Split channels
    vector<Mat> channels;
    split(ycrcb, channels);

    if (channels.size() != 3) {
        cout << "Error: Expected 3 channels after splitting YCrCb image." << endl;
        return Mat();
    }

    Mat y = channels[0];   // Y (luminance) channel
    Mat cr = channels[1];  // Cr channel  
    Mat cb = channels[2];  // Cb channel

    // Apply histogram equalization to Y channel
    Mat yEq;
    try {
        if (originalType == CV_8UC3) {
            // Y channel is already uint8, direct equalization
            equalizeHist(y, yEq);
        }
        else {
            // Convert Y channel to uint8 for equalization, then back
            Mat yUint8, yEqUint8;

            if (originalType == CV_32FC3) {
                // Normalized float to uint8
                y.convertTo(yUint8, CV_8UC1, 255.0);
            }
            else {
                // Other types
                y.convertTo(yUint8, CV_8UC1);
            }

            equalizeHist(yUint8, yEqUint8);

            // Convert back to original type
            if (originalType == CV_32FC3) {
                yEqUint8.convertTo(yEq, CV_32FC1, 1.0 / 255.0);
            }
            else {
                yEqUint8.convertTo(yEq, y.type());
            }
        }
    }
    catch (const Exception& e) {
        cout << "Warning: Error in histogram equalization, using original Y channel: "
            << e.what() << endl;
        yEq = y.clone();
    }

    // Merge channels back (Cr and Cb remain unchanged)
    vector<Mat> enhancedChannels = { yEq, cr, cb };
    Mat ycrcbEq;
    merge(enhancedChannels, ycrcbEq);

    // Convert back to BGR
    Mat enhancedBGR;
    try {
        enhancedBGR = customYCrCbtoBGR(ycrcbEq);
    }
    catch (const Exception& e) {
        cout << "Error converting back to BGR: " << e.what() << endl;
        return Mat();
    }

    return enhancedBGR;
}

// Alternative: Use OpenCV's built-in conversion for better performance and accuracy
Mat ContrastEnhancer::processOptimized(const Mat& imageBGR) {
    // Validate input
    if (!isValidInput(imageBGR)) {
        return Mat();
    }

    int originalType = imageBGR.type();
    Mat processedBGR = imageBGR;

    // Convert to uint8 if needed for color conversion
    if (originalType != CV_8UC3) {
        processedBGR = convertToUint8(imageBGR);
    }

    // Use OpenCV's optimized color conversion
    Mat ycrcb;
    cvtColor(processedBGR, ycrcb, COLOR_BGR2YCrCb);

    // Split channels
    vector<Mat> channels;
    split(ycrcb, channels);

    // Apply histogram equalization only to Y channel (already uint8)
    Mat yEq;
    equalizeHist(channels[0], yEq);

    // Merge back
    vector<Mat> enhancedChannels = { yEq, channels[1], channels[2] };
    Mat ycrcbEq;
    merge(enhancedChannels, ycrcbEq);

    // Convert back to BGR
    Mat enhancedBGR;
    cvtColor(ycrcbEq, enhancedBGR, COLOR_YCrCb2BGR);

    // Convert back to original type if needed
    if (originalType != CV_8UC3) {
        Mat result = convertFromUint8(enhancedBGR, originalType);
        return result;
    }

    return enhancedBGR;
}

int ContrastEnhancer::Demo(CStr imgDir, CStr outputDir) {
    // Create output directory if it doesn't exist
    CmFile::MkDir(outputDir);

    // Get all image files from input directory
    vecS names;
    string inDir, ext;
    int imgNum = CmFile::GetNamesNE(imgDir + "/*.jpg", names, inDir, ext);

    if (imgNum == 0) {
        printf("No images found in directory: %s\n", imgDir.c_str());
        return -1;
    }

    printf("Found %d images in %s\n", imgNum, inDir.c_str());
    printf("Processing contrast enhancement...\n");

    ContrastEnhancer enhancer;
    CmTimer timer("Contrast Enhancement");
    timer.Start();

    int successCount = 0;
    int failCount = 0;

    for (int i = 0; i < imgNum; i++) {
        string imageName = names[i] + ext;
        string inputPath = inDir + imageName;
        string outputPath = outputDir + "/" + names[i] + "_enhanced" + ext;
        string comparisonPath = outputDir + "/" + names[i] + "_comparison" + ext;

        printf("Processing %d/%d: %s\r", i + 1, imgNum, imageName.c_str());

        // Load original image
        Mat originalImg = imread(inputPath);
        if (originalImg.empty()) {
            printf("Error: Cannot load image %s\n", inputPath.c_str());
            failCount++;
            continue;
        }

        // Convert to float for processing (CV_32FC3)
        Mat originalImgF;
        originalImg.convertTo(originalImgF, CV_32FC3, 1.0 / 255.0);

        // Apply contrast enhancement
        Mat enhancedImgF = enhancer.process(originalImgF);
        if (enhancedImgF.empty()) {
            printf("Error: Contrast enhancement failed for %s\n", imageName.c_str());
            failCount++;
            continue;
        }

        // Convert back to uint8 for saving
        Mat enhancedImg;
        enhancedImgF.convertTo(enhancedImg, CV_8UC3, 255.0);

        // Save enhanced image
        if (!imwrite(outputPath, enhancedImg)) {
            printf("Error: Cannot save enhanced image %s\n", outputPath.c_str());
            failCount++;
            continue;
        }

        // Create side-by-side comparison
        Mat comparison;
        hconcat(originalImg, enhancedImg, comparison);

        // Add labels to comparison
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;
        Scalar color(0, 255, 0); // Green color

        putText(comparison, "Original", Point(10, 30), fontFace, fontScale, color, thickness);
        putText(comparison, "Enhanced", Point(originalImg.cols + 10, 30), fontFace, fontScale, color, thickness);

        // Save comparison image
        imwrite(comparisonPath, comparison);

        successCount++;
    }

    timer.Stop();

    printf("\nContrast Enhancement Demo Complete!\n");
    printf("Successfully processed: %d images\n", successCount);
    printf("Failed: %d images\n", failCount);
    printf("Average time per image: %.3f seconds\n", timer.TimeInSeconds() / max(successCount, 1));
    printf("Results saved to: %s\n", outputDir.c_str());

    // Generate statistics
    if (successCount > 0) {
        printf("\nGenerated files for each image:\n");
        printf("  - [name]_enhanced.jpg: Enhanced version of the original\n");
        printf("  - [name]_comparison.jpg: Side-by-side comparison\n");

        // Optional: Display first comparison if available
        if (successCount > 0) {
            string firstComparison = outputDir + "/" + names[0] + "_comparison.jpg";
            Mat sample = imread(firstComparison);
            if (!sample.empty()) {
                printf("\nPress any key to view sample comparison (first image)...\n");
                imshow("Contrast Enhancement Sample", sample);
                waitKey(0);
                destroyAllWindows();
            }
        }
    }

    return successCount;
}

// Alternative demo function that processes a single image
int ContrastEnhancer::DemoSingle(CStr inputImagePath, CStr outputDir) {
    CmFile::MkDir(outputDir);

    // Load image
    Mat originalImg = imread(inputImagePath);
    if (originalImg.empty()) {
        printf("Error: Cannot load image %s\n", inputImagePath.c_str());
        return -1;
    }

    printf("Processing single image: %s\n", inputImagePath.c_str());

    // Convert to float for processing
    Mat originalImgF;
    originalImg.convertTo(originalImgF, CV_32FC3, 1.0 / 255.0);

    ContrastEnhancer enhancer;
    CmTimer timer("Single Image Enhancement");
    timer.Start();

    // Apply contrast enhancement
    Mat enhancedImgF = enhancer.process(originalImgF);
    if (enhancedImgF.empty()) {
        printf("Error: Contrast enhancement failed\n");
        return -1;
    }

    timer.Stop();

    // Convert back to uint8
    Mat enhancedImg;
    enhancedImgF.convertTo(enhancedImg, CV_8UC3, 255.0);

    // Extract filename without extension
    string filename = CmFile::GetNameNE(inputImagePath);

    // Save results
    string enhancedPath = outputDir + "/" + filename + "_enhanced.jpg";
    string comparisonPath = outputDir + "/" + filename + "_comparison.jpg";

    imwrite(enhancedPath, enhancedImg);

    // Create comparison
    Mat comparison;
    hconcat(originalImg, enhancedImg, comparison);

    // Add labels
    putText(comparison, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    putText(comparison, "Enhanced", Point(originalImg.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

    imwrite(comparisonPath, comparison);

    printf("Processing completed in %.3f seconds\n", timer.TimeInSeconds());
    printf("Enhanced image saved to: %s\n", enhancedPath.c_str());
    printf("Comparison saved to: %s\n", comparisonPath.c_str());

    // Display results
    printf("Displaying results... Press any key to close.\n");
    imshow("Original vs Enhanced", comparison);
    waitKey(0);
    destroyAllWindows();

    return 1;
}