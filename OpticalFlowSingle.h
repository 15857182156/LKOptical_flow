//
// Created by wyl on 20-2-20.
//

#ifndef LKOPTICALFLOW_OPTICALFLOWSINGLE_H
#define LKOPTICALFLOW_OPTICALFLOWSINGLE_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class OpticalFlowSingle {
public:
    OpticalFlowSingle();
    ~OpticalFlowSingle();
    void calc(Mat &frame1, Mat &frame2, vector<KeyPoint> &vp1, vector<KeyPoint> &vp2, vector<bool> &tracked);
//    void compute_J_b(Mat &frame1, Mat &frame2, const KeyPoint &point1, const KeyPoint &point2, Mat &A, Mat &b);
    void compute_H_b(Mat &frame1, Mat &frame2, const KeyPoint &point1, const KeyPoint &point2, Mat &H, Mat &B, double &cost);
//    void runKernel(Mat &A, Mat &b, Mat& duv);
    int half_win = 4;
    int maxiter = 10;
    double th1 = 1;
};


inline float GetPixelValue(const Mat &img, float x, float y){
    uchar* data = &img.data[int(y) * img.step + int(x)];        // data就是指向矩阵数据的指针
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}



#endif //LKOPTICALFLOW_OPTICALFLOWSINGLE_H
