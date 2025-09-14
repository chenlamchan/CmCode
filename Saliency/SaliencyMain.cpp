#include "stdafx.h"
#include "../CmLib/Saliency/ContrastEnhancer.h"
#include "../CmLib/Saliency/CmAdaptiveTripleThresh.h"


int main(int argc, char* argv[])
{
    /*--------------For Saliency Test--------------*/
	if (argc != 2){
		printf("Usage: Saliency.exe wkDir\n");
		return 0;
	}

	//CStr wkDir = "D:/WkDir/Saliency/Test/";
	CStr wkDir = argv[1];
	CStr inDir = wkDir + "Imgs/", outDir = wkDir + "Saliency/";
	CmFile::Copy2Dir(inDir + "*.jpg", outDir);
	
	// Saliency detection method pretended in my ICCV 2013 paper http://mmcheng.net/effisalobj/.
	//CmSaliencyGC::Demo(inDir + "*.jpg", outDir); 

	// Saliency detection method presented in PAMI 2015 (CVPR 2011) paper http://mmcheng.net/salobj/.
	CmSalCut::Demo(inDir + "*.jpg", inDir + "*.png", outDir); //CmSaliencyRC::Get(inDir + "*.jpg", outDir);	

	//vecS des;
	//des.push_back("GC");  des.push_back("RC");
	//CmEvaluation::Evaluate(inDir + "*.png", outDir, wkDir + "Results.m", des);
	//CmEvaluation::EvalueMask(inDir + "*.png", outDir, "RCC", wkDir + "CutRes.m");



    ///*--------------For Contrast Enhancement Test--------------*/
    //if (argc != 2) {
    //    printf("Usage: ContrastEnhancer.exe wkDir\n");
    //    printf("Example: ContrastEnhancer.exe D:/WkDir/Contrast/\n");
    //    return 0;
    //}

    //// Following the same pattern as SaliencyMain.cpp
    //CStr wkDir = argv[1];
    //CStr inDir = wkDir + "Imgs/";
    //CStr outDir = wkDir + "Enhanced/";

    //printf("Working Directory: %s\n", wkDir.c_str());
    //printf("Input Directory: %s\n", inDir.c_str());
    //printf("Output Directory: %s\n", outDir.c_str());

    //// Copy original images to output directory (similar to SaliencyMain pattern)
    //CmFile::Copy2Dir(inDir + "*.jpg", outDir);

    //// Run contrast enhancement demo
    //int result = ContrastEnhancer::Demo(inDir, outDir);

    //if (result > 0) {
    //    printf("\nContrast Enhancement completed successfully!\n");
    //    printf("Processed %d images\n", result);
    //    printf("Results saved to: %s\n", outDir.c_str());
    //}
    //else {
    //    printf("\nContrast Enhancement failed or no images found.\n");
    //}

	return 0;
}
