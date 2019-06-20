//
// Created by aaronwd on 19-6-19.
//

#ifndef FACE_ALIGNMENT_FACE_ALIGN_H
#define FACE_ALIGNMENT_FACE_ALIGN_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class FaceAligner {
public:
    FaceAligner(int crop_size, string method);
    void warp_and_crop_face(cv::Mat& frame, Point2f local_pts[], Point2f facial_pts[]);

private:
    //initial parameter
    int p_crop_size;
    string p_method;
};

#endif //FACE_ALIGNMENT_FACE_ALIGN_H
