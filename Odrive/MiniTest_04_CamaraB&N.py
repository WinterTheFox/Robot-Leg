import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configurar la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
pipeline.start(config)
align = rs.align(rs.stream.color)

# ROI (área donde se espera el círculo rojo)
x_min, y_min = 400, 200
x_max, y_max = 550, 350
kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        # Obtener frames
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Obtener imágenes
        color_image = np.asanyarray(color_frame.get_data())
        roi_color = color_image[y_min:y_max, x_min:x_max]

        # Detectar color rojo en la ROI
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(roi_color, roi_color, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=100, param2=15, minRadius=7, maxRadius=50)

        z_val = float('nan')
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c
                mask_roi = mask[max(y - r, 0):min(y + r, mask.shape[0]),
                                max(x - r, 0):min(x + r, mask.shape[1])]
                if mask_roi.size == 0 or cv2.mean(mask_roi)[0] < 50:
                    continue
                x_full = x + x_min
                y_full = y + y_min
                z_val = depth_frame.get_distance(x_full, y_full)
                cv2.circle(color_image, (x_full, y_full), r, (0, 255, 0), 2)
                break

        # Dibujar ROI y mostrar Z
        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(color_image, f"Z: {1.42 - z_val:.3f} m", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.imshow("Color + Z", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
