#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <random>
#include <cstdlib>
#include "../src/ImageProc.hpp"

using namespace cv;

// Tiny assert helper
#define CHECK(cond) \
    do { if(!(cond)){ \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ \
                  << " -> " #cond << std::endl; std::exit(1);} } while(0)

// ---------------- RNG utilities (random per run, reproducible via SEED=...) ----------------
static std::mt19937& globalRng() {
    static std::mt19937 rng = []{
        if (const char* s = std::getenv("SEED")) {
            return std::mt19937(std::stoul(s));
        } else {
            std::random_device rd;
            std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd()};
            return std::mt19937(seq);
        }
    }();
    return rng;
}
static double urand(std::mt19937& rng, double a, double b) {
    std::uniform_real_distribution<double> d(a,b);
    return d(rng);
}
static bool bern(std::mt19937& rng, double p=0.5) {
    std::bernoulli_distribution d(p);
    return d(rng);
}

static cv::Scalar bgrFromHSV(int h, int s, int v) {
    cv::Mat hsv(1,1, CV_8UC3, cv::Scalar(h, s, v));
    cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b p = bgr.at<cv::Vec3b>(0,0);
    return cv::Scalar(p[0], p[1], p[2]); // B,G,R
}

// ---- Create a synthetic 3x3 marker image (BGR format) ----
static Mat createSyntheticMarkerBGR(int W=360, int H=360, int pad=20) {
    Mat img(H, W, CV_8UC3, Scalar(230,230,230));           // light background
    int cellW = (W-2*pad)/3, cellH = (H-2*pad)/3;

    std::vector<cv::Scalar> colors = {
        bgrFromHSV(  5,100,240), // Red
        bgrFromHSV( 60,220,240), // Green
        bgrFromHSV( 28,220,240), // Yellow
        bgrFromHSV(160,120,255), // Pink
        bgrFromHSV(150,180,255), // Magenta
        bgrFromHSV( 90,220,240), // Cyan
        bgrFromHSV(  0,180,240),
        bgrFromHSV( 60,220,240),
        bgrFromHSV( 28,220,240)
    };

    int k=0;
    for(int gy=0; gy<3; ++gy)
        for(int gx=0; gx<3; ++gx)
            rectangle(img,
                      Rect(pad+gx*cellW, pad+gy*cellH, cellW-4, cellH-4),
                      colors[k++], FILLED);

    rectangle(img, Rect(pad-6, pad-6, 3*cellW+8, 3*cellH+8), Scalar(20,20,20), 4);
    return img;
}

// ---- Geometry helpers ----
static Mat padReplicate(const Mat& src, int b=40){
    Mat out; copyMakeBorder(src, out, b, b, b, b, BORDER_REPLICATE);
    return out;
}

static Mat rotateDegrees(const Mat& src, double angle_deg){
    Point2f c(src.cols*0.5f, src.rows*0.5f);
    Mat M = getRotationMatrix2D(c, angle_deg, 1.0);
    Mat dst; warpAffine(src, dst, M, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// Perspective tilt (uniform shift on all sides)
static Mat perspectiveTilt(const Mat& src,
                           double offX_frac=0.08, double offY_frac=0.08) {
    int W = src.cols, H = src.rows;
    std::vector<Point2f> srcPts = {
        {0.f, 0.f}, {(float)W, 0.f}, {(float)W, (float)H}, {0.f, (float)H}
    };
    std::vector<Point2f> dstPts = {
        {(float)( W*offX_frac),          (float)( H*offY_frac)},
        {(float)( W*(1.0-offX_frac)),    (float)( H*offY_frac)},
        {(float)( W*(1.0-offX_frac)),    (float)( H*(1.0-offY_frac))},
        {(float)( W*offX_frac),          (float)( H*(1.0-offY_frac))}
    };
    Mat P = getPerspectiveTransform(srcPts, dstPts);
    Mat dst; warpPerspective(src, dst, P, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// Perspective twist with random displacement per corner
static Mat perspectiveTwistRandom(const Mat& src, double maxShiftFrac, std::mt19937& rng) {
    int W = src.cols, H = src.rows;
    auto shx = [&](double f){ return (float)urand(rng, -f, +f); };
    auto shy = [&](double f){ return (float)urand(rng, -f, +f); };
    std::vector<Point2f> srcPts = {
        {0.f, 0.f}, {(float)W, 0.f}, {(float)W, (float)H}, {0.f, (float)H}
    };
    std::vector<Point2f> dstPts = {
        { shx(W*maxShiftFrac),               shy(H*maxShiftFrac)               },
        { (float)W + shx(W*maxShiftFrac),    shy(H*maxShiftFrac)               },
        { (float)W + shx(W*maxShiftFrac),    (float)H + shy(H*maxShiftFrac)    },
        { shx(W*maxShiftFrac),               (float)H + shy(H*maxShiftFrac)    }
    };
    Mat P = getPerspectiveTransform(srcPts, dstPts);
    Mat dst; warpPerspective(src, dst, P, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// Random rotation
static Mat rotateRandom(const Mat& src, double max_deg, std::mt19937& rng){
    double a = urand(rng, -max_deg, +max_deg);
    Point2f c(src.cols*0.5f, src.rows*0.5f);
    Mat M = getRotationMatrix2D(c, a, 1.0);
    Mat dst; warpAffine(src, dst, M, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// Random tilt
static Mat perspectiveTiltRandom(const Mat& src, double maxOffX, double maxOffY, std::mt19937& rng){
    int W = src.cols, H = src.rows;
    double ox = urand(rng, -maxOffX, +maxOffX);
    double oy = urand(rng, -maxOffY, +maxOffY);
    std::vector<Point2f> srcPts = { {0,0},{(float)W,0},{(float)W,(float)H},{0,(float)H} };
    std::vector<Point2f> dstPts = {
        {(float)( W*ox),           (float)( H*oy)},
        {(float)( W*(1.0f+ox)),    (float)( H*oy)},
        {(float)( W*(1.0f+ox)),    (float)( H*(1.0f+oy))},
        {(float)( W*ox),           (float)( H*(1.0f+oy))}
    };
    Mat P = getPerspectiveTransform(srcPts, dstPts);
    Mat dst; warpPerspective(src, dst, P, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// ---- Add random Gaussian noise to the background ----
static void addBackgroundNoiseRandom(Mat& img, double sigma, std::mt19937& rng) {
    Mat noise(img.size(), CV_16SC3);
    std::normal_distribution<float> n(0.0f, (float)sigma);
    noise.forEach<Vec<short,3>>([&](Vec<short,3>& p, const int*){
        p[0] = (short)n(rng); p[1] = (short)n(rng); p[2] = (short)n(rng);
    });
    add(img, noise, img, noArray(), CV_8UC3);
}

// ---- Run the image processing pipeline ----
static bool runPipeline(const Mat& bgrInput, bool debug, int cell=24, int pad=3, int minColors=5) {
    ImageProcessor proc(debug);

    Mat bgr = bgrInput.clone();
    if (debug) { imshow("01 - Input BGR", bgr); waitKey(0); }

    Mat hsv = ImageProcessor::toHSV(bgr);
    if (debug) {
        Mat hsv_vis; cvtColor(hsv, hsv_vis, COLOR_HSV2BGR);
        imshow("02 - HSV visualized", hsv_vis); waitKey(0);
    }

    Mat mask = ImageProcessor::maskColors(hsv, {
        ImageProcessor::Color::Red, ImageProcessor::Color::Green,
        ImageProcessor::Color::Yellow, ImageProcessor::Color::Pink,
        ImageProcessor::Color::Magenta, ImageProcessor::Color::Cyan
    });
    if (debug) { imshow("03 - Color Mask (raw)", mask); waitKey(0); }

    mask = ImageProcessor::removeSmallBlobs(mask, 0.01);
    if (debug) { imshow("04 - After removeSmallBlobs", mask); waitKey(0); }

    mask = ImageProcessor::morphOpenClose(mask);
    if (debug) { imshow("05 - After morphOpenClose", mask); waitKey(0); }

    auto quad = ImageProcessor::findBestQuad(mask, bgr.size());
    Mat vis = bgr.clone();
    if (quad.has_value()) {
        for (int i=0;i<4;++i)
            line(vis, (*quad)[i], (*quad)[(i+1)&3], Scalar(0,255,0), 2);
    }
    if (debug) { imshow("06 - Detected Quad (overlay)", vis); waitKey(0); }
    if (!quad) return false;

    bool ok = proc.warpAndValidate3x3(bgr, *quad, cell, pad, minColors, debug);
    if (debug) {
        std::cout << "[DEBUG] warpAndValidate3x3 => " << (ok ? "valid" : "INVALID") << std::endl;
        waitKey(0);
        destroyAllWindows();
    }
    return ok;
}

int main() {
    std::cout << "Running unit test \n";

    {
        Mat bgr = createSyntheticMarkerBGR();

        // Add padding before transformations
        bgr = padReplicate(bgr, 60);

        // Shared RNG (different for each run unless SEED is set)
        auto& rng = globalRng();

        // Apply random rotation
        bgr = rotateRandom(bgr, /*max_deg=*/30.0, rng);

        // Randomly choose tilt or twist transformation
        if (bern(rng, 0.5)) {
            bgr = perspectiveTiltRandom(bgr, 0.06, 0.05, rng);
        } else {
            bgr = perspectiveTwistRandom(bgr,0.10, rng);
        }

        // Apply random background noise
        addBackgroundNoiseRandom(bgr, /*sigma=*/18.0, rng);

        // Resize back to maximum dimensions of 640x480
        bgr = ImageProcessor::resizeToMax(bgr, 640, 480);

        // Execute the pipeline
        bool ok = runPipeline(bgr, /*debug=*/false, 24, 3, 5);
        if (!ok) {
            std::cerr << "[INFO] Failed — rerunning with debug...\n";
            runPipeline(bgr, /*debug=*/true, 24, 3, 5);
        }
        CHECK(ok);
    }

    std::cout << "Test passed ✅\n";
    return 0;
}
