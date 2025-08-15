// main.cpp
// Usage: ./marker_coverage img1.jpg img2.png ...
//
// Prints per line:
//   <image_path> <coverage_percent>%
// or
//   <image_path> NO_MARKER
//   <image_path> LOAD_ERROR
//
// Extras:
//   - DEBUG=1   -> verbose step logs to stderr
//   - TIME=1    -> append total runtime

#include "ImageProc.hpp"
#include <iostream>
#include <chrono>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;

inline bool env_enabled(const char* key) {
    const char* v = std::getenv(key);
    return v && *v && std::string(v) != "0";
}
inline double ms_since(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
} // namespace

static void process_one(const std::string& path, ImageProcessor& proc) {
    const bool debug = env_enabled("DEBUG");
    const bool show_time = env_enabled("TIME");

    clk::time_point T0 = clk::now();

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << path << " LOAD_ERROR\n";
        return;
    }
    if (debug) {
        std::cerr << "\n=== Processing: " << path << " ===\n";
        std::cerr << "[INFO] Original size: " << img.cols << "x" << img.rows << "\n";
    }

    // Work at <= 640x480 max
    clk::time_point t0 = clk::now();
    cv::Mat small = ImageProcessor::resizeToMax(img, 640, 480);
    double scale = (double)img.cols / std::max(1, small.cols);
    clk::time_point t1 = clk::now();
    double ms_resize = ms_since(t0, t1);
    if (debug) {
        std::cerr << "[STEP] Resize: " << ms_resize << " ms  -> work size "
                  << small.cols << "x" << small.rows << " (scale=" << scale << ")\n";
    }

    // --- Preprocess ---
    t0 = clk::now();
    cv::Mat bgr = proc.denoiseBilateral(small, 5, 70.0, 7.0);
   // DBG_SHOW("Input BGR", bgr, 0);
    bgr = proc.grayWorldWB(bgr);
    bgr = proc.adaptiveGamma(bgr);
    bgr = proc.claheL(bgr, 2.0, {8,8});
    DBG_SHOW("After Cleaning", bgr, 0);
    t1 = clk::now();
    double ms_pre = ms_since(t0, t1);
    if (debug) std::cerr << "[STEP] Preprocess: " << ms_pre << " ms\n";

    // --- white prior ---
    t0 = clk::now();
    cv::Mat white = ImageProcessor::whiteMask(bgr);
    white = ImageProcessor::fillWhiteHoles(white, 5);
    DBG_SHOW("Remove White Background Stiker", white, 0);
    t1 = clk::now();
    double ms_white = ms_since(t0, t1);
    if (debug) std::cerr << "[STEP] White prior: " << ms_white << " ms\n";

    // --- Color mask in HSV ---
    t0 = clk::now();
    cv::Mat hsv = ImageProcessor::toHSV(bgr);
    cv::Mat mask = ImageProcessor::maskColors(
        hsv,
        { ImageProcessor::Color::Red,
          ImageProcessor::Color::Green,
          ImageProcessor::Color::Yellow,
          ImageProcessor::Color::Pink,
          ImageProcessor::Color::Magenta,
          ImageProcessor::Color::Cyan }
    );
    t1 = clk::now();
    double ms_mask = ms_since(t0, t1);
    if (debug) {
        std::cerr << "[STEP] HSV + Mask: " << ms_mask << " ms\n";
    }

    // --- Cleanup ---
    t0 = clk::now();
    bool micro = std::min(small.cols, small.rows) < 100;
    double minAreaFrac = micro ? 0.005 : 0.01;         // keep small markers on tiny images
    cv::Mat m2 = ImageProcessor::removeSmallBlobs(mask, 0.01);
    DBG_SHOW("Small Blobs Removed", m2, 0);
    cv::Mat m3;
    if (micro) {
        cv::Mat k5 = cv::getStructuringElement(cv::MORPH_RECT, {5,5});
        cv::morphologyEx(m2, m3, cv::MORPH_CLOSE, k5, {-1,-1}, 1);
        m3 = ImageProcessor::morphOpenClose(m3);
    } else {
        m3 = ImageProcessor::morphOpenClose(m2);
    }
    t1 = clk::now();
    double ms_clean = ms_since(t0, t1);
    if (debug) {
        std::cerr << "[STEP] Cleanup: " << ms_clean << " ms\n";
        int nz = cv::countNonZero(m3);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(m3.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::cerr << "[INFO] Mask NZ=" << nz << ", contours=" << contours.size() << "\n";
    }
    DBG_SHOW("Morphology Op", m3, 0);

    // --- Find best quadrilateral ---
    t0 = clk::now();
    auto quadOpt = ImageProcessor::findBestQuad(m3, small.size());
    t1 = clk::now();
    double ms_quad = ms_since(t0, t1);
    if (debug) std::cerr << "[STEP] Find Quad: " << ms_quad << " ms\n";

    if (!quadOpt) {
        double total_ms = ms_since(T0, clk::now());
        if (debug) std::cerr << "[RESULT] NO_MARKER (total " << total_ms << " ms)\n";
        if (show_time) std::cout << path << " NO_MARKER (ms=" << total_ms << ")\n";
        else           std::cout << path << " NO_MARKER\n";
        return;
    }

    // --- Validate by content (3x3 colored grid) ---
    const auto& quad = *quadOpt;

    t0 = clk::now();
    bool ok = proc.warpAndValidate3x3(bgr, quad, /*cell*/24, /*pad*/3, /*minDistinctColors*/5, /*debug*/false);
    t1 = clk::now();
    double ms_valid = ms_since(t0, t1);
    if (debug) std::cerr << "[STEP] Validate 3x3: " << ms_valid << " ms\n";

    if (!ok) {
        double total_ms = ms_since(T0, clk::now());
        if (debug) std::cerr << "[RESULT] NO_MARKER (failed content validation, total "
                             << total_ms << " ms)\n";
        if (show_time) std::cout << path << " NO_MARKER (ms=" << total_ms << ")\n";
        else           std::cout << path << " NO_MARKER\n";
        return;
    }

    // --- Coverage on original image coordinates ---
    t0 = clk::now();
    int cov = ImageProcessor::coveragePercent(img.size(), quad, scale);
    t1 = clk::now();
    double ms_cov = ms_since(t0, t1);
    if (debug) std::cerr << "[STEP] Coverage%: " << ms_cov << " ms\n";

    // draw
    cv::Mat vis = img.clone();
    for (int i=0;i<4;++i)
        cv::line(vis, quad[i]*scale, quad[(i+1)&3]*scale, {0,255,0}, 2);
    DBG_SHOW("quad", vis, 0);

    double total_ms = ms_since(T0, clk::now());
    if (debug) {
        std::cerr << "[TOTAL] " << total_ms << " ms  (resize:" << ms_resize
                  << ", pre:" << ms_pre
                  << ", white:" << ms_white
                  << ", mask:" << ms_mask
                  << ", clean:" << ms_clean
                  << ", quad:" << ms_quad
                  << ", valid:" << ms_valid
                  << ", cov:" << ms_cov
                  << ")\n";
    }

    if (show_time) std::cout << path << " " << cov << "% (ms=" << total_ms << ")\n";
    else           std::cout << path << " " << cov << "%\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << (argc>0?argv[0]:"marker_coverage") << " <image1> [image2 ...]\n";
        return 1;
    }

    ImageProcessor proc(false); // debug off; set true for verbose visual debugging

    for (int i = 1; i < argc; ++i) {
        process_one(argv[i], proc);
    }
    return 0;
}
