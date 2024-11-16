import cv2 as cv
import matplotlib.pyplot as plt
from util import format_path
import numpy as np

class ImageProcessing:
    def __init__(self):
        pass

    def cv2_trackbar(self, image_path):
        image_path = format_path(image_path)
        image = cv.imread(image_path)
        image = cv.resize(image, (500, 500))
        
        def nothing(x):
            pass

        # Create a properly sized window
        cv.namedWindow("Color Filter", cv.WINDOW_NORMAL)
        cv.resizeWindow("Color Filter", 800, 600)

        # Create trackbars for RGB
        cv.createTrackbar("R", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("G", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("B", "Color Filter", 255, 255, nothing)

        # Create trackbars for HSV
        cv.createTrackbar("Hue", "Color Filter", 179, 179, nothing)
        cv.createTrackbar("Saturation", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("Value", "Color Filter", 255, 255, nothing)

        # Create filter type selector
        cv.createTrackbar("Filter (0:RGB 1:HSV)", "Color Filter", 0, 1, nothing)

        while True:
            # Get RGB values
            r = cv.getTrackbarPos("R", "Color Filter")
            g = cv.getTrackbarPos("G", "Color Filter")
            b = cv.getTrackbarPos("B", "Color Filter")

            # Get HSV values
            h = cv.getTrackbarPos("Hue", "Color Filter")
            s = cv.getTrackbarPos("Saturation", "Color Filter")
            v = cv.getTrackbarPos("Value", "Color Filter")

            # Get filter type
            filter_type = cv.getTrackbarPos("Filter (0:RGB 1:HSV)", "Color Filter")

            if filter_type == 0:  # RGB filtering
                lower = np.array([b, g, r])
                upper = np.array([255, 255, 255])
                mask = cv.inRange(image, lower, upper)
            else:  # HSV filtering
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                lower = np.array([h, s, v])
                upper = np.array([179, 255, 255])
                mask = cv.inRange(hsv, lower, upper)

            result = cv.bitwise_and(image, image, mask=mask)
            
            # Stack images horizontally with proper spacing
            display = np.hstack((image, result))
            
            # Show result
            cv.imshow("Color Filter", display)

            if cv.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cv.destroyAllWindows()

    def area_of_contour(self, contour):
        return cv.contourArea(contour)


    def crop_board(self, image_path, board_contour) -> np.ndarray:
        image_path = format_path(image_path)

        # Load image
        image = cv.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Get bounding box coordinates of the largest contour
        x, y, w, h = cv.boundingRect(board_contour)

        # Crop the image using the bounding box coordinates
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image


    def _find_board_contour(self, image_path):
        image_path = format_path(image_path)

        # Load image
        image = cv.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # filter hue above 70
        lower_bound = np.array([146, 179, 149])
        upper_bound = np.array([255, 255, 255])

        # Create masks based on the bounds
        mask = cv.inRange(image, lower_bound, upper_bound)

        # Apply the masks to the images
        result = cv.bitwise_and(image, image, mask=mask)
        image = result

        # Convert image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply GaussianBlur
        blur = cv.GaussianBlur(gray, (7, 7), 0)

        # Apply Canny edge detection
        edges = cv.Canny(blur, 60, 80, apertureSize=3)
        
        # Create a kernel for dilation
        kernel = np.ones((5,5), np.uint8)
        
        # Dilate the edges to connect nearby contours
        dilated_edges = cv.dilate(edges, kernel, iterations=2)

        # Find contours on the dilated edges
        contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Find the largest contour by area
        largest_contour = max(contours, key=cv.contourArea)

        return largest_contour
    

    def find_largest_contour(self, image: np.ndarray) -> tuple:
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply GaussianBlur
        blur = cv.GaussianBlur(gray, (7, 7), 0)

        # Apply Canny edge detection
        edges = cv.Canny(blur, 60, 80, apertureSize=3)
        
        # Create a kernel for dilation
        kernel = np.ones((5,5), np.uint8)
        
        # Dilate the edges to connect nearby contours
        dilated_edges = cv.dilate(edges, kernel, iterations=2)

        # Find contours on the dilated edges
        contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # For visualization
        # debug_image = cv.cvtColor(image.copy() if len(image.shape) == 3 else cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR), cv.COLOR_BGR2RGB)
        # cv.drawContours(debug_image, contours, -1, (0, 255, 0), 3)
        # plt.imshow(debug_image)
        # plt.show()

        if not contours:  # If no contours found
            return None, None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv.contourArea)

        # Get the bounding rectangle
        x, y, w, h = cv.boundingRect(largest_contour)
        rect = np.array([x, y, x+w, y+h])

        return largest_contour, rect
    
    def draw_rect(self, image: np.ndarray, rect: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Draw rectangle on image
        Args:
            image: Input image
            rect: Rectangle coordinates [x1, y1, x2, y2]
            color: Color in BGR format
            thickness: Line thickness
        Returns:
            Image with drawn rectangle
        """
        if rect is not None:
            x1, y1, x2, y2 = rect
            return cv.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)
        return image


    def find_difference_between_images(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        if image1 is None:
            raise FileNotFoundError(f"Image not found at {image1}")
        if image2 is None:
            raise FileNotFoundError(f"Image not found at {image2}")
        

        # Convert images to HSV
        # hsv1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
        # hsv2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)

        # Define the lower and upper bounds for hue and value
        lower_bound = np.array([146, 179, 149])
        upper_bound = np.array([255, 255, 255])

        # Create masks based on the bounds
        mask1 = cv.inRange(image1, lower_bound, upper_bound)
        mask2 = cv.inRange(image2, lower_bound, upper_bound)

        # Apply the masks to the images
        filtered_image1 = cv.bitwise_and(image1, image1, mask=mask1)
        filtered_image2 = cv.bitwise_and(image2, image2, mask=mask2)

        # Use the filtered images for further processing
        image1 = filtered_image1
        image2 = filtered_image2


        # Convert images to grayscale
        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        # debug_image1 = gray1.copy()
        # debug_image1= cv.resize(debug_image1, (800, 800))
        # debug_image2 = gray2.copy()
        # debug_image2 = cv.resize(debug_image2, (800, 800))
        # cv.imshow("Image 1", debug_image1)
        # cv.imshow("Image 2", debug_image2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


        # Resize images to the same size
        gray1 = cv.resize(gray1, (gray2.shape[1], gray2.shape[0]))

        # Apply Gaussian blur to reduce noise
        gausian_kernel = (3, 3)
        gray1 = cv.GaussianBlur(gray1, gausian_kernel, 0)
        gray2 = cv.GaussianBlur(gray2, gausian_kernel, 0)

        # Compute the absolute difference between the two images
        difference = cv.absdiff(gray1, gray2)
        
        # Apply a higher threshold to eliminate more noise
        _, difference = cv.threshold(difference, 70, 255, cv.THRESH_BINARY)

        # # Clean up noise
        kernel = np.ones((2, 2), np.uint8)

        difference = cv.erode(difference, kernel, iterations=2)
        difference = cv.dilate(difference, kernel, iterations=1)

        return difference



if __name__ == "__main__":
    image_processing = ImageProcessing()
    image_path = "../data/train/game_1/1_12.jpg"
    board_contour = image_processing._find_board_contour(image_path)
    contour_image = cv.drawContours(cv.imread(image_path), [board_contour], -1, (0, 255, 0), 3)
    # # Get bounding box coordinates of the largest contour
    # x, y, w, h = cv.boundingRect(board_contour)

    # # Crop the image using the bounding box coordinates
    # cropped_image = cv.imread(image_path)[y:y+h, x:x+w]

    # # Display the cropped image
    # plt.figure()
    # plt.title("Cropped Image")
    # plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
    # plt.show()

    # plt.figure()
    # plt.title("Largest Contour Only")
    # plt.imshow(cv.cvtColor(contour_image, cv.COLOR_BGR2RGB))
    # plt.show()
    image_processing.cv2_trackbar(image_path)