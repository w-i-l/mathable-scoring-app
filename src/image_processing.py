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
        lower_bound = np.array([0, 39, 149])
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
        dilated_edges = cv.erode(dilated_edges, kernel, iterations=1)

        # Find contours on the dilated edges
        contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Find the largest contour by area
        largest_contour = max(contours, key=cv.contourArea)
        
        # Get the center of the largest contour
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2

        # Create a square contour of 1985x1985
        half_size = 1470 // 2
        square_contour = np.array([
            [[center_x - half_size, center_y - half_size]],  # Top-left
            [[center_x + half_size, center_y - half_size]],  # Top-right
            [[center_x + half_size, center_y + half_size]],  # Bottom-right
            [[center_x - half_size, center_y + half_size]]   # Bottom-left
        ], dtype=np.int32)

        return square_contour
    

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
        kernel = np.ones((3,3), np.uint8)
        
        # Dilate the edges to connect nearby contours
        dilated_edges = cv.dilate(edges, kernel, iterations=2)
        dilated_edges = cv.erode(dilated_edges, kernel, iterations=2)

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
        _, difference = cv.threshold(difference, 60, 255, cv.THRESH_BINARY)

        # # Clean up noise
        kernel = np.ones((3, 3), np.uint8)

        difference = cv.erode(difference, kernel, iterations=2)
        difference = cv.dilate(difference, kernel, iterations=1)


        return difference
    

    def find_contous_centroid(self, contours):
        moments = cv.moments(contours)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return cX, cY
    

    def split_board_in_blocks(self, board: np.ndarray) -> tuple:
        """ Divide the board into 14x14 blocks, distributing remaining pixels evenly
        
        Args:
            board: Input board image
            
        Returns:
            tuple: (Image with grid drawn, list of rectangle coordinates)
        """
        # Create a copy of the board for drawing
        result = board.copy()
        
        # Get board dimensions
        height, width = board.shape[:2]
        
        # Calculate base block size and remaining pixels
        base_block_height = height // 14
        base_block_width = width // 14
        
        remaining_height = height % 14
        remaining_width = width % 14
        
        # Calculate how many blocks will get an extra pixel
        # We'll distribute from the edges inward
        extra_height_blocks = remaining_height // 2  # blocks from top and bottom
        extra_width_blocks = remaining_width // 2    # blocks from left and right
        
        # Lists to store block dimensions
        block_heights = []
        block_widths = []
        
        # Calculate heights for each row
        for i in range(14):
            if i < extra_height_blocks or i >= 14 - extra_height_blocks:
                # Outer blocks get an extra pixel
                block_heights.append(base_block_height + 1)
            else:
                block_heights.append(base_block_height)
                
        # Calculate widths for each column
        for j in range(14):
            if j < extra_width_blocks or j >= 14 - extra_width_blocks:
                # Outer blocks get an extra pixel
                block_widths.append(base_block_width + 1)
            else:
                block_widths.append(base_block_width)
                
        # Calculate cumulative positions
        y_positions = [0]
        for h in block_heights:
            y_positions.append(y_positions[-1] + h)
            
        x_positions = [0]
        for w in block_widths:
            x_positions.append(x_positions[-1] + w)
        
        # Generate and draw rectangles
        rectangles = []
        for i in range(14):
            for j in range(14):
                # Get block coordinates
                x1 = x_positions[j]
                y1 = y_positions[i]
                x2 = x_positions[j + 1]
                y2 = y_positions[i + 1]
                
                # Store rectangle coordinates
                rectangles.append(((x1, y1), (x2, y2)))
                
                # Draw rectangle
                cv.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Optional: Add block coordinates for debugging
                # cv.putText(result, f"{i},{j}", (x1 + 5, y1 + 20),
                #           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result, rectangles




if __name__ == "__main__":
    image_processing = ImageProcessing()
    i = 4
    board_1 = cv.imread(f"../data/cropped/board_{i}.jpg")
    board_2 = cv.imread(f"../data/cropped/board_{i+1}.jpg")
    
    ################
    # image_processing.cv2_trackbar(f"../data/cropped/board_{i}.jpg")
    # exit(0)
    # contour = image_processing._find_board_contour(f"../data/cropped/board_{i}.jpg")
    # image = cv.imread(f"../data/cropped/board_{i}.jpg")
    # image = cv.drawContours(image, [contour], -1, (0, 255, 0), 3)
    # image = cv.resize(image, (800, 800))
    # cv.imshow("Image", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # exit(0)
    ################

    diff = image_processing.find_difference_between_images(board_1, board_2)
    diff_piece = image_processing.find_largest_contour(diff)

    contours = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    debig_diff = diff.copy()
    debig_diff = cv.cvtColor(debig_diff, cv.COLOR_GRAY2BGR)
    cv.drawContours(debig_diff, contours[0], -1, (0, 255, 0), 3)
    debig_diff = cv.resize(debig_diff, (800, 800))
    cv.imshow("Difference", debig_diff)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Get original dimensions before any resizing
    original_height, original_width = diff.shape[:2]
    
    x, y, w, h = cv.boundingRect(diff_piece[0])
    w = 105
    h = 105
    
    # Scale coordinates for board_2
    scale_x = board_2.shape[1] / original_width
    scale_y = board_2.shape[0] / original_height
    
    board_contour = [
        int(x * scale_x), 
        int(y * scale_y), 
        int((x + w) * scale_x), 
        int((y + h) * scale_y)
    ]
    
    # Get grid blocks
    board_2, grid_rectangles = image_processing.split_board_in_blocks(board_2)
    
    # Find matching block
    matching_block_idx, overlap_percentage = find_matching_block(board_contour, grid_rectangles)
    
    if matching_block_idx != -1:
        # Draw matching block in different color
        matched_rect = grid_rectangles[matching_block_idx]
        cv.rectangle(board_2, matched_rect[0], matched_rect[1], (0, 0, 255), 2)  # Red color for matched block
        
        # Draw the contour rectangle
        board_2 = image_processing.draw_rect(board_2, board_contour, color=(0, 255, 0), thickness=2)  # Green color for contour
        
    board_2 = cv.resize(board_2, (800, 800))
    diff = cv.resize(diff, (800, 800))
    cv.imshow("Board 2", board_2)
    cv.imshow("Difference", diff)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # image_path = "../data/cropped_board.jpg"
    # board_contour = image_processing._find_board_contour(image_path)
    # contour_image = cv.drawContours(cv.imread(image_path), [board_contour], -1, (0, 255, 0), 3)
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
    # image_processing.cv2_trackbar(image_path)