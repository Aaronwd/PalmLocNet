//
// Created by aaronwd on 19-6-19.
//
#include <iostream>
#include <math.h>

#include "face_align.h"

#define PI acos(-1)

FaceAligner::FaceAligner(int crop_size, string method)
{
    p_crop_size = crop_size;
    p_method = method;
}

void FaceAligner::warp_and_crop_face(cv::Mat& frame, Point2f local_pts[], Point2f facial_pts[])
{
    if (FaceAligner::p_method == "scaling")
    {
        cout << "face alignment method: " << FaceAligner::p_method<< endl;
        Point2f dstp[] = {
                Point2f(0.0, 0.0),
                Point2f((float) p_crop_size, 0.0),
                Point2f((float) p_crop_size, (float) p_crop_size)
        };
        Mat tfm = cv::getAffineTransform(local_pts, dstp);
        cv::warpAffine(frame, frame, tfm, Size(p_crop_size, p_crop_size));
        cout << "size: " << frame.size() << endl;
        cout << "new size: " << Size(224,224) << endl;
    }
    else if (FaceAligner::p_method == "rotation_translation")
    {
        cout << "face alignment method: " << FaceAligner::p_method<< endl;
        float maxX = frame.size().width;
        float minX = 0;
        float maxY = frame.size().height;
        float minY = 0;

        for (int i = 0; i < 3; i++) {
            if (local_pts[i].x > maxX) {
                local_pts[i].x = maxX;
            }
            if (local_pts[i].x < minX) {
                local_pts[i].x = minX;
            }
            if (local_pts[i].y > maxY) {
                local_pts[i].y = maxY;
            }
            if (local_pts[i].y < minY) {
                local_pts[i].y = minY;
            }
        }

        frame = frame(Rect(local_pts[0].x, local_pts[0].y,local_pts[2].x, local_pts[2].y));
        float maxsize = max(frame.size().width, frame.size().height);
        float scale = maxsize/p_crop_size;

        int a = (int)(frame.size().width/scale);
        int b = (int)(frame.size().height/scale);

        cv::resize(frame, frame, Size(b, a));
        Point2f left_eye = (facial_pts[0]-local_pts[0])/scale;
        Point2f right_eye = (facial_pts[1]-local_pts[0])/scale;

//  ################### translation ##########################
    Point2f ground_eyes_center(56.0, 45.0);
    Point2f truth_eyes_center = (left_eye+right_eye)/2;
    Point2f t = ground_eyes_center-truth_eyes_center;
    Mat translation_matrix = (Mat_<float>(2,3)<<1,0,t.x, 0,1,t.y);
    cv::warpAffine(frame, frame, translation_matrix, Size(p_crop_size, p_crop_size));

//##################### rotation #############################
    double theta = atan2(right_eye.y-left_eye.y, right_eye.x-left_eye.x);
    theta = theta/PI*180;
    Mat rotation_matrix = cv::getRotationMatrix2D(ground_eyes_center, theta, 1.0);
    cv::warpAffine(frame, frame, rotation_matrix, Size(p_crop_size, p_crop_size));
    }
}

