//
// Created by wyl on 20-2-20.
//

#include "OpticalFlowSingle.h"

OpticalFlowSingle::OpticalFlowSingle() {}

OpticalFlowSingle::~OpticalFlowSingle() {}

//void OpticalFlowSingle::compute_J_b(Mat &frame1, Mat &frame2, const KeyPoint &point1, const KeyPoint &point2, Mat &A, Mat &b) {
//    double Mat_A[9][2], Mat_b[9];
//    Mat tempA(9, 2, CV_64F, Mat_A);
//    Mat tempb(9, 1, CV_64F, Mat_b);
//    float x1 = point1.pt.x;
//    float y1 = point1.pt.y;
//    float x2 = point2.pt.x;
//    float y2 = point2.pt.y;
//    int n = 0;
//    for(int j = -4; j < 5; j++) {
//        for(int i = -4; i < 5; i++) {
//            Mat_A[n][0] = GetPixelValue(frame2, x2 + i, y2 + j) - GetPixelValue(frame2, x2 + i - 1, y2 + j);
//            Mat_A[n][1] = GetPixelValue(frame2, x2 + i, y2 + j) - GetPixelValue(frame2, x2 + i, y2 + j - 1);
//            Mat_b[n] = GetPixelValue(frame1, x1 + i, y1 + j) - GetPixelValue(frame2, x2 + i, y2 + j);
//            n++;
//        }
//    }
//    A = tempA.clone();
//    b = tempb.clone();
//}


void OpticalFlowSingle::compute_H_b(Mat &frame1, Mat &frame2, const KeyPoint &point1, const KeyPoint &point2, Mat &H,
                                    Mat &B, double &cost) {
    double J[2], err;
    Mat tempJ(1, 2, CV_64F, &J);
    Mat temperr(1, 1, CV_64F, &err);
    Mat tempH(2, 2, CV_64F, Scalar::all(0));
    Mat tempB(2, 1, CV_64F, Scalar::all(0));
    float x1 = point1.pt.x;
    float y1 = point1.pt.y;
    float x2 = point2.pt.x;
    float y2 = point2.pt.y;
    int n = 0;

    for(int j = -half_win; j < half_win; j++) {
        for(int i = -half_win; i < half_win; i++) {
            J[0] = GetPixelValue(frame2, x2 + i, y2 + j) - GetPixelValue(frame2, x2 + i - 1, y2 + j);
            J[1] = GetPixelValue(frame2, x2 + i, y2 + j) - GetPixelValue(frame2, x2 + i, y2 + j - 1);
            err = GetPixelValue(frame1, x1 + i, y1 + j) - GetPixelValue(frame2, x2 + i, y2 + j);

            tempH += tempJ.t() * tempJ;
            tempB = tempJ.t() * temperr;

            B += tempB;
            cost += err * err / 2;
            n++;
        }
    }
    H = tempH.clone();

}

//void OpticalFlowSingle::runKernel(Mat &A, Mat &b, Mat& duv) {
//    Mat ATA(2, 2, CV_64F, Scalar::all(0));
//    Mat g(2, 1, CV_64F, Scalar::all(0));
//    ATA = A.t() * A;
//    gemm(A, b, 1, noArray(), 0, g, GEMM_1_T);
//    duv = -1 * ATA.inv() * g;
//    cout<< "duv:"<< duv.at<double>(0) <<'\t'<< duv.at<double>(1) << endl;
//}


void OpticalFlowSingle::calc(Mat &frame1, Mat &frame2, vector<KeyPoint> &vp1, vector<KeyPoint> &vp2, vector<bool> &tracked) {
    // parameters
    int count = (int)vp1.size();

    for(int k = 0; k < count; k++){
        if(!tracked[k])
            continue;

        double lastCost = 0;
        for(int iter = 0; iter < maxiter; iter++){
            Mat H(2, 2, CV_64F, Scalar::all(0));
            Mat B(2, 1, CV_64F, Scalar::all(0));
            Mat duv(2, 1, CV_64F, Scalar::all(0));
            if(vp1[k].pt.x <half_win || vp1[k].pt.y <half_win || vp1[k].pt.x > frame1.cols - half_win || vp1[k].pt.y > frame1.rows - half_win) {
                break;
            }
            if(vp2[k].pt.x <half_win || vp2[k].pt.y <half_win || vp2[k].pt.x > frame1.cols - half_win || vp2[k].pt.y > frame1.rows - half_win){
                break;
            }

            double cost = 0;
            compute_H_b(frame1, frame2, vp1[k], vp2[k], H, B, cost);
            duv = H.inv() * B;

            /** 截止条件 **/
            if (std::isnan(duv.at<double>(0))) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                tracked[k] = false;
                cout << "update is nan" << endl;
                break;
            }

            if (iter > 0 && cost > lastCost) {
//                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // 更新角点
            vp2[k].pt.x += duv.at<double>(0);
            vp2[k].pt.y += duv.at<double>(1);

            lastCost = cost;
        }
    }
}