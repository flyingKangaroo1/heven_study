#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage

class LidarCameraProjection:
    def __init__(self):
        rospy.init_node("lidar_projection_visualizer", anonymous=False)

        # ------------------------------------------------
        # 1. Calibration Matrices (Copied from your code)
        # ------------------------------------------------
        
        # Camera Intrinsic Parameters
        self.intrinsic_matrix_left = np.array([[683.4524,    0.0,     318.4096],
                                               [  0.0,     685.3216, 240.1933],
                                               [  0.0,        0.0,        1.0]], dtype=np.float64)

        ### 빈칸을 채우세요 ###
        self.intrinsic_matrix_right = np.array([[],
                                                [],
                                                []], dtype=np.float64)
        
        # Lidar -> Camera Extrinsic Parameters (3x4 [R|t])
        self.extrinsic_matrix_left = np.array([[ 0.5032, -0.8642, -0.0011,  0.0925],
                                               [-0.2176, -0.1254, -0.9679,  0.2056],
                                               [ 0.8364,  0.4873, -0.2512,  0.0061]], dtype=np.float64)

        ### 빈칸을 채우세요 ###
        self.extrinsic_matrix_right = np.array([[],
                                                [],
                                                []], dtype=np.float64)

        # ------------------------------------------------
        # 2. Data Holders
        # ------------------------------------------------
        self.left_image = None
        self.right_image = None

        # ------------------------------------------------
        # 3. Subscribers
        # ------------------------------------------------
        # Changed queue_size to 1 to reduce lag for visualization
        self.lidar_sub = rospy.Subscriber("/velodyne_points/noground", PointCloud2, self.lidar_callback, queue_size=1)
        self.left_image_sub = rospy.Subscriber("/usb_cam_Left/image_raw/compressed", CompressedImage, self.left_camera_callback, queue_size=1)
        self.right_image_sub = rospy.Subscriber("/usb_cam_Right/image_raw/compressed", CompressedImage, self.right_camera_callback, queue_size=1)

        rospy.loginfo("Projection Visualizer Initialized. Waiting for topics...")

    def left_camera_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.left_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def right_camera_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.right_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def lidar_callback(self, msg):
        # Ensure we have images to draw on
        if self.left_image is None or self.right_image is None:
            return

        # 1. Convert ROS PointCloud2 to List of [x, y, z]
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if not points:
            return

        # 2. Create copies of images to draw on
        vis_left = self.left_image.copy()
        vis_right = self.right_image.copy()

        # 3. Project Points
        self.project_and_draw(points, vis_left, self.intrinsic_matrix_left, self.extrinsic_matrix_left, "Left")
        self.project_and_draw(points, vis_right, self.intrinsic_matrix_right, self.extrinsic_matrix_right, "Right")

        # 4. Show Images
        cv2.imshow("Lidar Projection - Left", vis_left)
        cv2.imshow("Lidar Projection - Right", vis_right)
        cv2.waitKey(1)

    def project_and_draw(self, points, image, intrinsic, extrinsic, name_tag):
        """
        Projects 3D points onto the 2D image and draws circles.
        """
        height, width = image.shape[:2]
        
        for p in points:
            # Create Homogeneous World Coordinate [x, y, z, 1]
            world_point = np.array([p[0], p[1], p[2], 1.0])

            # 1. Extrinsic Transform: World -> Camera Coordinate
            camera_point = extrinsic @ world_point

            # 2. Intrinsic Transform: Camera -> Image Plane (Homogeneous)
            # We only need the first 3 elements of camera_point for multiplication
            projected_point = intrinsic @ camera_point[:3]

            z_depth = projected_point[2]

            # Check if point is in front of the camera
            if z_depth > 0:
                # Perspective Division (Normalization)
                u = int(projected_point[0] / z_depth)
                v = int(projected_point[1] / z_depth)

                # Check if point is inside image boundaries
                if 0 <= u < width and 0 <= v < height:
                    # Color based on depth (optional visualization logic)
                    # Close points = Green, Far points = Blue
                    dist = np.sqrt(p[0]**2 + p[1]**2)
                    color_val = max(0, min(255, int(255 - (dist * 10))))
                    color = (255, color_val, 0) # Blue-Green gradient

                    cv2.circle(image, (u, v), 2, color, -1)

if __name__ == "__main__":
    try:
        node = LidarCameraProjection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
