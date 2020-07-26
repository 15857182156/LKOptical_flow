//
// Created by wyl on 20-2-20.
//
#include "OpticalFlowSingle.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

void lkimshow(Mat& now_frame, Mat& mask, vector<KeyPoint> &vp1, vector<KeyPoint> &vp2, vector<bool> &tracked){
    Mat img;
    cv::cvtColor(now_frame, img, COLOR_GRAY2BGR);
    add(img, mask, img);
    for (int i = 0; i < vp1.size(); i++) {
        if(tracked[i] == false){
            continue;
        }
        cv::circle(img, vp2[i].pt, 2, cv::Scalar(0, 250, 0), 2);
//        cv::circle(img, vp1[i].pt, 2, cv::Scalar(0, 0, 255), 2);
        cv::line(img, vp1[i].pt, vp2[i].pt, cv::Scalar(0, 250, 0));
    }
    cv::imshow("LK", img);
    cv::waitKey(1);
}


void MultiLK(Ptr<OpticalFlowSingle> &Of, Mat old_frame, Mat now_frame, vector<KeyPoint> &vp1, vector<KeyPoint> &vp2, vector<bool> &tracked){
    // 构建金字塔图片层
    const float p[4] = {0.125, 0.25, 0.5, 1};
    vector<Mat> pyr_oldimg, pyr_nowimg;
    vector<vector<KeyPoint>> pyr_kp;
    for(size_t i=0; i<4; i++){
        Mat img1, img2;
        Size dsize = Size(int(p[i] * old_frame.cols), int(p[i] * old_frame.rows));          // width, height
        resize(old_frame, img1, dsize, 0, 0, INTER_LINEAR);
        resize(now_frame, img2, dsize, 0, 0, INTER_LINEAR);
        pyr_oldimg.push_back(img1);
        pyr_nowimg.push_back(img2);
    }

    for(size_t i = 0; i < vp1.size(); i++){
        vp1[i].pt.x = vp1[i].pt.x/8;
        vp1[i].pt.y = vp1[i].pt.y/8;
        vp2[i].pt.x = vp2[i].pt.x/8;
        vp2[i].pt.y = vp2[i].pt.y/8;
    }
    Of->calc(pyr_oldimg[0], pyr_nowimg[0], vp1, vp2, tracked);

    for(size_t i = 0; i < vp1.size(); i++){
        vp1[i].pt.x *= 2;
        vp1[i].pt.y *= 2;
        vp2[i].pt.x *= 2;
        vp2[i].pt.y *= 2;
    }
    Of->calc(pyr_oldimg[1], pyr_nowimg[1], vp1, vp2, tracked);

    for(size_t i = 0; i < vp1.size(); i++){
        vp1[i].pt.x *= 2;
        vp1[i].pt.y *= 2;
        vp2[i].pt.x *= 2;
        vp2[i].pt.y *= 2;
    }
    Of->calc(pyr_oldimg[2], pyr_nowimg[2], vp1, vp2, tracked);

    for(size_t i = 0; i < vp1.size(); i++){
        vp1[i].pt.x *= 2;
        vp1[i].pt.y *= 2;
        vp2[i].pt.x *= 2;
        vp2[i].pt.y *= 2;
    }
    Of->calc(pyr_oldimg[2], pyr_nowimg[2], vp1, vp2, tracked);
}

void SingleLK(Ptr<OpticalFlowSingle> &Of, Mat frame1, Mat frame2, vector<KeyPoint> &vp1, vector<KeyPoint> &vp2, vector<bool> &tracked){
    Of->calc(frame1, frame2, vp1, vp2, tracked);
}

int main(){
    bool single = 0;

    string filename = "/home/wyl/桌面/视频/slow_traffic_small.mp4";

    VideoCapture capture(filename);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    Mat old_frame, now_frame;
    capture >> old_frame;

    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);


    vector<KeyPoint>  vp1, vp2;

    while(true){
        capture >> now_frame;
        cvtColor(now_frame, now_frame, COLOR_BGR2GRAY);

        // 提取特征点
        if(vp1.size() < 1){
            Ptr<GFTTDetector> detector =  GFTTDetector::create(500, 0.01, 20, 3, false, 0.04); // maximum 500 keypoints
            detector->detect(old_frame, vp1);

//            Ptr<FeatureDetector> detector = ORB::create(100);
//            detector->detect ( old_frame ,vp1 );

//            Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(60);
//            detector->detect(old_frame, vp1);
        }
        cout << "the number of points：" << vp1.size() << endl;

        // vp2初始值
        vp2.assign(vp1.begin(), vp1.end());
        vector<bool>  tracked(vp1.size(), true);

        Ptr<OpticalFlowSingle> Of = new OpticalFlowSingle();

        if (single){
            SingleLK(Of, old_frame, now_frame, vp1, vp2, tracked);
        }else{
            MultiLK(Of, old_frame, now_frame, vp1, vp2, tracked);
        }

        // 更新vp, 判断是否越界
        vector<KeyPoint> tempvp;
        for(size_t i = 0; i < vp1.size(); i++){
            if(vp2[i].pt.x > now_frame.cols-7 || vp2[i].pt.y > now_frame.rows-7 || vp2[i].pt.x < 7 || vp2[i].pt.y < 7)
                tracked[i] = false;
            if(tracked[i])
                tempvp.push_back(vp2[i]);
        }

        // mask
        for(size_t i = 0; i < vp1.size(); i++){
            circle(mask, vp1[i].pt, 1, Scalar(0,255,255), 2);
        }

        lkimshow(now_frame, mask, vp1, vp2, tracked);

        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // 更新
        vp1 = tempvp;
        old_frame = now_frame;
    }
}