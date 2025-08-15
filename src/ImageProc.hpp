//  ImageProc.hpp
//  marker_coverage
//  Created by Harry Cohen on 11/08/2025
//
//  ImageProcessor: small toolkit to detect a colored 3x3 marker,
//  find its quad, validate it by color, and report coverage.

#ifndef ImageProc_hpp
#define ImageProc_hpp

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>
#include <array>
#include <set>
#include <iostream>

#ifdef DEBUG_IMGPROC
    #define DBG_SHOW(name, img, wait) \
        do { cv::imshow(name, img); cv::waitKey(wait); } while(0)
#else
    #define DBG_SHOW(name, img, wait) \
        do { } while(0)
#endif


class ImageProcessor {
public:
    // Marker palette (HSV-based thresholds are defined in hsvRanges)
    enum class Color { Red, Green, Yellow, Pink, Magenta, Cyan };

    explicit ImageProcessor(bool debug=false);  // enable extra logs/visuals

    // -------- I/O --------
    // List PNG/JPEG files under a folder (non-recursive).
    std::vector<std::string> listImages(const std::string& folder) const;
    // Pick one random image from a folder (returns empty Mat if none).
    cv::Mat loadRandomImage(const std::string& folder) const;

    // -------- Preprocess --------
    // Keep aspect ratio, clamp to max width/height.
    static cv::Mat resizeToMax(const cv::Mat& img, int maxW=640, int maxH=480);
    // BGR → HSV helper.
    static cv::Mat toHSV(const cv::Mat& bgr);
    
    // -------- White prior (optional) --------
    // Heuristic mask for bright/neutral regions (HSV + Lab).
    static cv::Mat whiteMask(const cv::Mat& bgr,
                             int lMin=210, int sMax=60, int vMin=200, int aMax=18, int bMax=18);
    // Close small holes inside the white mask.
    static cv::Mat fillWhiteHoles(const cv::Mat& white, int closeK=5);

    // -------- Color masks --------
    // Single-color mask from HSV using pre-defined ranges.
    static cv::Mat maskColor(const cv::Mat& hsv, Color c);
    // OR-combine several color masks.
    static cv::Mat maskColors(const cv::Mat& hsv, const std::vector<Color>& colors);

    // -------- Image utilities --------
    // Quick viewer (debug only). Pass wait<0 to skip waitKey.
    static void show(const std::string& title, const cv::Mat& img, int wait=0);

    // Edge-preserving denoise (bilateral).
    cv::Mat denoiseBilateral(const cv::Mat& bgr,
                             int d = 7,
                             double sigmaColor = 70.0,
                             double sigmaSpace = 5.0);
    // Simple gray-world white balance.
    cv::Mat grayWorldWB(const cv::Mat& bgr);
    // Adaptive gamma using median luminance (Lab).
    cv::Mat adaptiveGamma(const cv::Mat& bgr,
                          int thresh = 128,
                          double gamma_dark = 1.2,
                          double gamma_bright = 0.9);
    // CLAHE on luminance (Lab) for local contrast.
    cv::Mat claheL(const cv::Mat& bgr, double clipLimit = 2.0, cv::Size tiles = {8,8});
    // Open+Close to remove speckles and fill small gaps.
    static cv::Mat morphOpenClose(const cv::Mat& bin);
    // Drop connected components below a fraction of the image area.
    static cv::Mat removeSmallBlobs(const cv::Mat& bin, double minAreaFrac);

    // -------- Quad detection --------
    // From a binary mask, find the best quadrilateral covering the marker.
    static std::optional<std::array<cv::Point2f,4>>
    findBestQuad(const cv::Mat& bin, cv::Size imgSize);

    // -------- Warp & validate (3x3 grid) --------
    // Perspective-warp to a 3x3 board; count distinct colors to validate.
    bool warpAndValidate3x3(const cv::Mat& bgr,
                            const std::array<cv::Point2f,4>& quad,
                            int cell = 24,      // side of each cell in the warped image
                            int pad  = 3,       // small sampling pad around cell centers
                            int minDistinctColors = 6,
                            bool debug = false) const;

    // -------- Coverage (%) --------
    // Compute % area of the quad over the original image.
    // Provide the quad at "small" scale and the scale factor to original.
    static int coveragePercent(cv::Size orig,
                               const std::array<cv::Point2f,4>& quad_at_small,
                               double scaleSmallToOrig);

    // (Optional helpers; kept public if you want to debug them)
    static cv::Mat colorfulMaskBGR(const cv::Mat& bgr, double chromaMin = 0.12);
    static cv::Mat brightWhiteMaskHSV(const cv::Mat& hsv, int vMinWhite = 200, int sMaxWhite = 40);

private:
    bool debug_;  // toggles extra logs/visualization

    // HSV thresholds per color (filled by hsvRanges).
    static void hsvRanges(Color c,
                          std::vector<cv::Scalar>& lowers,
                          std::vector<cv::Scalar>& uppers);

    // Order quad points clockwise: TL, TR, BR, BL.
    static std::array<cv::Point2f,4> orderQuadCW(std::array<cv::Point2f,4> q);
    // Signed polygon area (positive for CW ordering).
    static double polygonArea(const std::array<cv::Point2f,4>& q);

    // Classify a small HSV patch into one of the marker colors.
    int classifyPatchHSV(const cv::Vec3b& hsv) const;
};

#endif /* ImageProc_hpp */
