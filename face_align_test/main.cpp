//
// Created by aaronwd on 19-6-19.
//

#include "face_align.h"

int main() {
    int crop_size = 112;
    string method = "scaling";
//    string method = "rotation_translation";
    FaceAligner f(crop_size, method);
    string inputImage = "/home/aaronwd/桌面/face-test/f001/2.jpg";
    cv::Mat img = cv::imread(inputImage);
    Point2f b[] = {
            Point2f(50, 50),
            Point2f(250.0, 0.0),
            Point2f(250.0, 300.0)
    };
    cout << b[0]<< b[1]<< b[2] << endl;

        Point2f facial[] = {
                Point2f(50.0, 50.0),
                Point2f(50.0, 400.0),
                Point2f(400.0, 400.0)
        };

        f.warp_and_crop_face(img, b, facial);
        cout << b[0]<< b[1]<< b[2] << endl;
        cv::imshow("show", img);
        waitKey(100);
        return 0;

}