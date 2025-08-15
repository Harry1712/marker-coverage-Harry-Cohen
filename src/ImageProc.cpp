//
// ImageProc.cpp
// marker_coverage
//
// Created by Harry Cohen on 11/08/2025
//
// Implementation of ImageProcessor for color-marker detection.

#include "ImageProc.hpp"
#include <filesystem>
#include <random>
#include <set>
using namespace cv;
namespace fs = std::filesystem;

ImageProcessor::ImageProcessor(bool debug) : debug_(debug) {}


// List all PNG/JPEG files in a given folder
std::vector<std::string> ImageProcessor::listImages(const std::string& folder) const {
    std::vector<std::string> files;
    if (!fs::exists(folder)) return files;

    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        for (auto& ch : ext) ch = std::tolower(ch);
        if (ext == ".png" || ext == ".jpeg") {
            files.push_back(e.path().string());
        }
    }
    if (debug_) std::cerr << "[listImages] found " << files.size() << " files in " << folder << "\n";
    return files;
}

// Load a random image from a folder
cv::Mat ImageProcessor::loadRandomImage(const std::string& folder) const {
    auto files = listImages(folder);
    if (files.empty()) return {};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, (int)files.size()-1);
    auto path = files[dist(gen)];

    if (debug_) std::cerr << "[loadRandomImage] " << path << "\n";
    return cv::imread(path, cv::IMREAD_COLOR);
}

// Resize while keeping aspect ratio, max dimensions given
cv::Mat ImageProcessor::resizeToMax(const cv::Mat& img, int maxW, int maxH) {
    if (img.empty()) return img;
    if (img.cols <= maxW && img.rows <= maxH) return img;

    double sx = (double)maxW / img.cols;
    double sy = (double)maxH / img.rows;
    double s  = std::min(sx, sy);

    cv::Mat out;
    cv::resize(img, out, cv::Size(), s, s, cv::INTER_AREA);
    return out;
}

// Convert BGR to HSV
cv::Mat ImageProcessor::toHSV(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

// Bilateral filter to smooth but keep edges
cv::Mat ImageProcessor::denoiseBilateral(const cv::Mat& bgr,
                                         int d,
                                         double sigmaColor,
                                         double sigmaSpace)
{
    if (bgr.empty()) return bgr;
    cv::Mat out;
    cv::bilateralFilter(bgr, out, d, sigmaColor, sigmaSpace);
    if (debug_) {
        std::cerr << "[denoiseBilateral] d=" << d
                  << " sigmaColor=" << sigmaColor
                  << " sigmaSpace=" << sigmaSpace << "\n";
    }
    return out;
}

// Gray-world white balance
cv::Mat ImageProcessor::grayWorldWB(const cv::Mat& bgr) {
    CV_Assert(!bgr.empty() && bgr.type()==CV_8UC3);
    cv::Mat out = bgr.clone();
    cv::Scalar m = cv::mean(out);                   // moyennes B,G,R
    double g = (m[0] + m[1] + m[2]) / 3.0 + 1e-9;   // niveau gris visé
    out.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int*) {
        for (int c = 0; c < 3; ++c) {
            double v = p[c] * (g / (m[c] + 1e-9));
            p[c] = cv::saturate_cast<uchar>(v);
        }
    });
    return out;
}

// Helper: compute median L channel value (Lab space)
static int medianL(const cv::Mat& bgr) {
    cv::Mat lab; cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch); // ch[0] = L
    int hist[256] = {0};
    for (int y = 0; y < ch[0].rows; ++y) {
        const uchar* r = ch[0].ptr<uchar>(y);
        for (int x = 0; x < ch[0].cols; ++x) hist[r[x]]++;
    }
    int tot = ch[0].rows * ch[0].cols, acc = 0;
    for (int i = 0; i < 256; ++i) { acc += hist[i]; if (acc >= tot/2) return i; }
    return 128;
}

// Adaptive gamma correction based on brightness
cv::Mat ImageProcessor::adaptiveGamma(const cv::Mat& bgr,
                      int thresh ,
                      double gamma_dark ,
                      double gamma_bright) {
    CV_Assert(!bgr.empty() && bgr.type()==CV_8UC3);
    int med = medianL(bgr);
    double gamma = (med < thresh) ? gamma_dark : gamma_bright;

    cv::Mat f32, out;
    bgr.convertTo(f32, CV_32F, 1.0/255.0);
    cv::pow(f32, gamma, f32);
    f32.convertTo(out, CV_8U, 255.0);
    return out;
}

// CLAHE on luminance channel for local contrast enhancement
cv::Mat ImageProcessor::claheL(const cv::Mat& bgr, double clipLimit, cv::Size tiles) {
    CV_Assert(!bgr.empty() && bgr.type()==CV_8UC3);
    cv::Mat lab; cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch);      // ch[0] = L
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tiles);
    clahe->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::Mat out; cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

void ImageProcessor::hsvRanges(Color c,
                               std::vector<cv::Scalar>& lowers,
                               std::vector<cv::Scalar>& uppers) {
    lowers.clear(); uppers.clear();
    switch (c) {
        case Color::Red:
            lowers = { {0, 100, 70},  {170, 100, 70} };
            uppers = { {10,200,255},  {180,200,255} };
            break;
        case Color::Green:
            lowers = { {40,  40, 40} }; uppers = { {85, 255,255} };
            break;
        case Color::Yellow:
            lowers = { {18, 50, 30} }; uppers = { {40, 255,255} };
            break;
        case Color::Pink:
            lowers = { {145,30, 180} }; uppers = { {178,140,255} };
            break;
        case Color::Magenta:
            lowers = { {110, 40, 35}, {145, 80, 120} }; uppers = { {160, 200, 255},{150, 200, 255} };
            break;
        case Color::Cyan:
            lowers = { {70, 50, 100} }; uppers = { {95,255,255} };
            break;
    }
}

// Create mask for a given color, with noise cleanup
cv::Mat ImageProcessor::maskColor(const cv::Mat& hsv, Color c) {
    std::vector<cv::Scalar> L, U;
    hsvRanges(c, L, U);

    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8U);
    for (size_t i = 0; i < L.size(); ++i) {
        cv::Mat m;
        cv::inRange(hsv, L[i], U[i], m);
        cv::bitwise_or(mask, m, mask);
    }

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    return mask;
}

// Morph open+close to remove noise and fill gaps
cv::Mat ImageProcessor::morphOpenClose(const cv::Mat& bin) {
    CV_Assert(bin.type()==CV_8U);
    cv::Mat m = bin.clone();
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    cv::morphologyEx(m, m, cv::MORPH_OPEN,  k, {-1,-1}, 1); // enlève bruit
    cv::morphologyEx(m, m, cv::MORPH_CLOSE, k, {-1,-1}, 2); // bouche trous
    return m;
}

// Remove blobs smaller than a given fraction of image area
cv::Mat ImageProcessor::removeSmallBlobs(const cv::Mat& bin, double minAreaFrac) {
    CV_Assert(bin.type()==CV_8U);
    double minA = minAreaFrac * bin.total();
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat out = cv::Mat::zeros(bin.size(), CV_8U);
    for (auto& c : cs) {
        if (std::fabs(cv::contourArea(c)) >= minA) {
            cv::drawContours(out, std::vector<std::vector<cv::Point>>{c}, -1, 255, cv::FILLED);
        }
    }
    return out;
}

// Find best quadrilateral from contours
std::optional<std::array<cv::Point2f,4>>
ImageProcessor::findBestQuad(const cv::Mat& bin, cv::Size imgSize) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double imgA = (double)imgSize.width * imgSize.height;
    double bestScore = 0.0;
    std::optional<std::array<cv::Point2f,4>> best;

    for (auto& c : cs) {
        double a = std::fabs(cv::contourArea(c));
        if (a < 0.005 * imgA || a > 0.98 * imgA) continue;

        std::vector<cv::Point> hull;
        cv::convexHull(c, hull);
        double ah = std::fabs(cv::contourArea(hull));
        double solidity = (ah > 1e-6) ? (a / ah) : 0.0;
        if (solidity < 0.60) continue;

        double peri = cv::arcLength(hull, true);
        std::vector<cv::Point> poly;
        cv::approxPolyDP(hull, poly, 0.05 * peri, true);

        std::array<cv::Point2f,4> q;
        if (poly.size() == 4) {
            for (int i=0;i<4;++i) q[i] = poly[i];
        } else if (poly.size() >= 5 && poly.size() <= 6) {
            cv::RotatedRect rr = cv::minAreaRect(poly);
            cv::Point2f pts[4]; rr.points(pts);
            for (int i=0;i<4;++i) q[i] = pts[i];
        } else {
            continue;
        }
        q = orderQuadCW(q);

        double score = a / (peri*peri + 1e-6) + 0.000001 * a;
        if (score > bestScore) { bestScore = score; best = q; }
    }
    return best;
}


// Warp quad to 3x3 grid and check if enough distinct colors are found
bool ImageProcessor::warpAndValidate3x3(const cv::Mat& bgr,
                                        const std::array<cv::Point2f,4>& quad,
                                        int cell, int pad,
                                        int minDistinctColors,
                                        bool debug) const
{
    const int W = 3*cell, H = 3*cell;
    std::array<cv::Point2f,4> q = orderQuadCW(quad);
    std::vector<cv::Point2f> src = { q[0], q[1], q[2], q[3] };
    std::vector<cv::Point2f> dst = { {0,0},{(float)W,0},{(float)W,(float)H},{0,(float)H} };

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warp; cv::warpPerspective(bgr, warp, M, {W,H}, cv::INTER_LINEAR);
    cv::Mat hsv; cv::cvtColor(warp, hsv, cv::COLOR_BGR2HSV);

    std::set<int> labels;
    for (int gy=0; gy<3; ++gy) {
        for (int gx=0; gx<3; ++gx) {
            int cx = gx*cell + cell/2;
            int cy = gy*cell + cell/2;

            int x0 = std::max(0, cx - pad), x1 = std::min(W-1, cx + pad);
            int y0 = std::max(0, cy - pad), y1 = std::min(H-1, cy + pad);

            int cnt=0, sumH=0, sumS=0, sumV=0;
            for (int y=y0; y<=y1; ++y) {
                const cv::Vec3b* r = hsv.ptr<cv::Vec3b>(y);
                for (int x=x0; x<=x1; ++x) {
                    sumH += r[x][0]; sumS += r[x][1]; sumV += r[x][2]; cnt++;
                }
            }
            cv::Vec3b avg( (uchar)(sumH/cnt), (uchar)(sumS/cnt), (uchar)(sumV/cnt) );
            int lab = classifyPatchHSV(avg);
            if (lab >= 0) labels.insert(lab);
        }
    }

    if (debug) {
        cv::Mat vis = warp.clone();
        for (int i=1;i<3;++i) {
            cv::line(vis, {i*cell,0}, {i*cell,H-1}, {0,255,0}, 1);
            cv::line(vis, {0,i*cell}, {W-1,i*cell}, {0,255,0}, 1);
        }
        cv::imshow("warp-3x3", vis); cv::waitKey(1);
    }

    return (int)labels.size() >= minDistinctColors;
}

// Classify a patch color by checking HSV ranges
int ImageProcessor::classifyPatchHSV(const cv::Vec3b& hsv) const {
    auto hit = [&](Color c)->bool{
        std::vector<cv::Scalar> L,U; hsvRanges(c, L, U);
        for (size_t i=0;i<L.size();++i) {
            bool okH = (hsv[0]>=L[i][0] && hsv[0]<=U[i][0]) || (L[i][0]>U[i][0] && (hsv[0]>=L[i][0]||hsv[0]<=U[i][0]));
            if (okH && hsv[1]>=L[i][1] && hsv[1]<=U[i][1] && hsv[2]>=L[i][2] && hsv[2]<=U[i][2])
                return true;
        }
        return false;
    };
    if (hit(Color::Red))     return 0;
    if (hit(Color::Green))   return 1;
    if (hit(Color::Yellow))  return 2;
    if (hit(Color::Pink))    return 3;
    if (hit(Color::Magenta)) return 4;
    if (hit(Color::Cyan))    return 5;
    return -1;
}

// Compute % coverage of quad over original image
int ImageProcessor::coveragePercent(cv::Size orig,
                                    const std::array<cv::Point2f,4>& quad_at_small,
                                    double scaleSmallToOrig)
{
    // remet la quad à l’échelle de l’image d’origine
    std::array<cv::Point2f,4> Q = quad_at_small;
    for (auto& p : Q) { p.x *= (float)scaleSmallToOrig; p.y *= (float)scaleSmallToOrig; }

    double A = std::fabs(polygonArea(Q));
    double total = (double)orig.width * orig.height;
    int cov = (int)std::lround(100.0 * A / std::max(1.0, total));
    return std::clamp(cov, 0, 100);
}

// Order quad points clockwise starting from top-left
std::array<cv::Point2f,4> ImageProcessor::orderQuadCW(std::array<cv::Point2f,4> q) {
    // order: top-left, top-right, bottom-right, bottom-left (CW)
    std::sort(q.begin(), q.end(), [](auto& a, auto& b){
        if (a.y == b.y) return a.x < b.x;
        return a.y < b.y;
    });
    cv::Point2f tl = q[0], tr = q[1], bl = q[2], br = q[3];
    if (tr.x < tl.x) std::swap(tl, tr);
    if (br.x < bl.x) std::swap(bl, br);
    return { tl, tr, br, bl };
}

// Compute area of a 4-point polygon
double ImageProcessor::polygonArea(const std::array<cv::Point2f,4>& q) {
    double a=0;
    for (int i=0;i<4;++i) {
        const cv::Point2f& p = q[i];
        const cv::Point2f& n = q[(i+1)&3];
        a += p.x * n.y - p.y * n.x;
    }
    return 0.5 * a;
}

// Combine masks of multiple colors
cv::Mat ImageProcessor::maskColors(const cv::Mat& hsv, const std::vector<Color>& colors) {
    cv::Mat maskTotal = cv::Mat::zeros(hsv.size(), CV_8U);
    for (auto c : colors) {
        cv::Mat m = maskColor(hsv, c);
        cv::bitwise_or(maskTotal, m, maskTotal);
    }
    return maskTotal;
}

// Show image in a window (for debug)
void ImageProcessor::show(const std::string& title, const cv::Mat& img, int wait) {
    cv::imshow(title, img);
    if (wait >= 0) cv::waitKey(wait);
}

// Detect white areas (used for prior)
cv::Mat ImageProcessor::whiteMask(const cv::Mat& bgr, int lMin, int sMax, int vMin, int aMax, int bMax){
    CV_Assert(bgr.type()==CV_8UC3);
    cv::Mat hsv; cvtColor(bgr,hsv,cv::COLOR_BGR2HSV);
    cv::Mat mHSV; inRange(hsv, Scalar(0,0,vMin), Scalar(180,sMax,255), mHSV);
    cv::Mat lab; cvtColor(bgr,lab,COLOR_BGR2Lab); std::vector<Mat> ch; split(lab,ch);
    cv::Mat mL,mA,mB; inRange(ch[0], lMin, 255, mL);
    inRange(ch[1], 128-aMax, 128+aMax, mA);
    inRange(ch[2], 128-bMax, 128+bMax, mB);
    cv::Mat white = mHSV | (mL & mA & mB);
    cv::Mat k = getStructuringElement(cv::MORPH_RECT,{3,3});
    morphologyEx(white,white,cv::MORPH_OPEN,k);
    morphologyEx(white,white,cv::MORPH_CLOSE,k);
    return white;
}

// Fill holes inside white areas
Mat ImageProcessor::fillWhiteHoles(const cv::Mat& white, int closeK){
    Mat k = getStructuringElement(MORPH_RECT,{closeK,closeK});
    Mat filled; morphologyEx(white, filled, MORPH_CLOSE, k, {-1,-1}, 2);
    return filled;
}

