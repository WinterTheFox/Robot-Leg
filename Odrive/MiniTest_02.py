import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configurar el flujo de RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

# Alinear imágenes
align_to = rs.stream.color
align = rs.align(align_to)

# Coordenadas del rectángulo de interés (ROI)
x_min, y_min = 400, 200
x_max, y_max = 550, 350

pipeline.start(config)
print("🎥 Cámara iniciada. Presiona 'q' para salir.")

# Kernel para morfología
kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        frameset = pipeline.wait_for_frames()
        aligned_frames = align.process(frameset)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        roi_color = color_image[y_min:y_max, x_min:x_max]

        # Convertir a HSV y detectar rojo con rango mejorado
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Operaciones morfológicas para limpiar la máscara
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Aplicar máscara a ROI y convertir a gris
        result = cv2.bitwise_and(roi_color, roi_color, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Suavizar con GaussianBlur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detectar círculos con parámetros ajustados
        circles = cv2.HoughCircles(blurred,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=30,
                                   param1=100,  # detector de bordes Canny más estricto
                                   param2=15,   # umbral más bajo para detección
                                   minRadius=7,
                                   maxRadius=50)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c
                # Verificar que el círculo esté bien cubierto por la máscara (evitar falsas detecciones)
                mask_roi = mask[max(y - r, 0):min(y + r, mask.shape[0]),
                                max(x - r, 0):min(x + r, mask.shape[1])]
                if mask_roi.size == 0:
                    continue
                mean_val = cv2.mean(mask_roi)[0]
                if mean_val < 50:  # umbral mínimo en la máscara
                    continue

                x_full = x + x_min
                y_full = y + y_min
                z = depth_frame.get_distance(x_full, y_full)
                cv2.circle(color_image, (x_full, y_full), r, (0, 255, 0), 2)
                cv2.circle(color_image, (x_full, y_full), 2, (0, 0, 255), 3)
                cv2.putText(color_image, f"Z: {z:.3f} m", (x_full + 10, y_full - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Solo mostrar el primer círculo detectado
                break

        cv2.imshow("RealSense + Detección Círculo Rojo Mejorada", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.001)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 Cámara detenida.")
