#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>

using namespace cv;

// Function to determine color name based on HSV values
std::string detectColor(cv::Scalar hsv) {
    int h = static_cast<int>(hsv[0]);
    int s = static_cast<int>(hsv[1]);
    int v = static_cast<int>(hsv[2]);

    // Define HSV ranges for different colors
    if (s < 40) {
        return "Gray";
    }
    if (v < 40) {
        return "Black";
    }
    if (h >= 0 && h <= 25) {
        return "Red";
    }
    if (h > 25 && h <= 55) {
        return "Yellow";
    }
    if (h > 55 && h <= 85) {
        return "Green";
    }
    if (h > 85 && h <= 150) {
        return "Blue";
    }
    if (h > 150 && h <= 320) {
        return "Red";
    }
    return "Unknown";
}

void identifyShapes(Mat& src) {
    Mat gray, blurred, edges;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;

 
    // Apply Gaussian blur
    GaussianBlur(src, blurred, Size(3, 3), 1.5);

    // Detect edges using Canny
    Canny(blurred, edges, 100, 200);

    // Find contours
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw contours on the central box
    for (size_t i = 0; i < contours.size(); i++) {
        // Approximate contour to a polygon
        std::vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        
        // Calculate the contour area and the perimeter
        double area = contourArea(contours[i]);
        double perimeter = arcLength(contours[i], true);

        // Calculate the circularity (4 * pi * area) / (perimeter * perimeter)
        double circularity = (4 * M_PI * area) / (perimeter * perimeter);

        // Identify shape based on circularity and number of vertices
        std::string shape;
        std::string part;

        // Consider circularity close to 1 as a circle
        if (circularity > 0.9) {
            shape = "Circle";
            part = "Good part";
        } else {
            int vertices = (int)approx.size();
            if (vertices == 3) shape = "Triangle", part = "Good part";
            else if (vertices == 4) shape = "Square", part = "Good part";
            else if (vertices == 6) shape = "Hexagon", part = "Good part";
            else shape = "No shape detected", part = "Bad part";
        }
        // Print shape information to terminal
        printf("Detected shape: %s\n", shape.c_str());
        printf("Good or bad part? %s\n", part.c_str());
        // Draw the name of the shape on the central box
        putText(src, shape, approx[0], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    }
}

int main()
{
    // Open the video camera.
    std::string pipeline = "libcamerasrc"
        " ! video/x-raw, width=800, height=600" // camera needs to capture at a higher resolution
        " ! videoconvert"
        " ! videoscale"
        " ! video/x-raw, width=400, height=300" // can downsample the image after capturing
        " ! videoflip method=rotate-180" // remove this line if the image is upside-down
        " ! appsink drop=true max_buffers=2";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
        printf("Could not open camera.\n");
        return 1;
    }

    // Create the OpenCV window
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;

    // Measure the frame rate - initialise variables
    int frame_id = 0;
    timeval start, end;
    gettimeofday(&start, NULL);

    // Variable to keep track of the last time color and shape detection occurred
    double last_detection_time = 0;

    for(;;) {
        if (!cap.read(frame)) {
            printf("Could not read a frame.\n");
            break;
        }
        
        // Show frame
        cv::imshow("Camera", frame);
        cv::waitKey(1);

        // Get the frame dimensions
        int height = frame.rows;
        int width = frame.cols;

        // Define the size of the central box (e.g., 100x100 pixels)
        int boxSize = 200;
        // Calculate the coordinates of the central box
        int startX = (width - boxSize) / 2;
        int startY = (height - boxSize) / 2;

        // Define the region of interest (ROI) for the central box
        cv::Rect roi(startX, startY, boxSize, boxSize);

        // Crop the central box from the frame
        cv::Mat centralBox = frame(roi);

        // Convert central box to HSV
        cv::Mat hsvBox;
        cv::cvtColor(centralBox, hsvBox, cv::COLOR_BGR2HSV);

        // Create masks for white and black
        cv::Mat whiteMask, blackMask;
        inRange(hsvBox, Scalar(0, 0, 200), Scalar(180, 40, 255), whiteMask);
        inRange(hsvBox, Scalar(0, 0, 0), Scalar(180, 255, 50), blackMask);

        // Combine masks
        cv::Mat combinedMask;
        bitwise_or(whiteMask, blackMask, combinedMask);

        // Invert the mask to exclude white and black pixels
        cv::Mat nonWhiteBlackMask;
        bitwise_not(combinedMask, nonWhiteBlackMask);

        // Apply the mask to the HSV image
        cv::Mat filteredHSV;
        hsvBox.copyTo(filteredHSV, nonWhiteBlackMask);

        // Calculate mean HSV value (excluding white and black pixels)
        cv::Scalar meanHSV = cv::mean(filteredHSV, nonWhiteBlackMask);

        // Get current time
        gettimeofday(&end, NULL);
        double current_time = end.tv_sec + end.tv_usec / 1000000.0;

        // Perform color and shape detection once every second
        if (current_time - last_detection_time >= 1.0) {
            // Detect color
            std::string colorName = detectColor(meanHSV);

            // Print color name to console
            std::cout << "Detected color: " << colorName << std::endl;

            // Process the central box for shape detection
            identifyShapes(centralBox);

            // Update last detection time
            last_detection_time = current_time;
        }

        // Display the central box for visualization (optional)
        cv::imshow("Central Box", centralBox);

        // Measure the frame rate
        frame_id++;
        if (frame_id >= 30) {
            gettimeofday(&end, NULL);
            double diff = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec)/1000000.0;
            printf("30 frames in %f seconds = %f FPS\n", diff, 30/diff);
            frame_id = 0;
            gettimeofday(&start, NULL);
        }
    }

    // Free the camera
    cap.release();
    return 0;
}
