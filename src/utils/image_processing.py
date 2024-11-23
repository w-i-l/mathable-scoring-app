import cv2 as cv
import matplotlib.pyplot as plt
from .helper_functions import format_path
import numpy as np


class ImageProcessing:

    def cv2_trackbar(self, image_path):
        image_path = format_path(image_path)
        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv.resize(image, (500, 500))
        
        def nothing(x):
            pass

        # create a properly sized window
        cv.namedWindow("Color Filter", cv.WINDOW_NORMAL)
        cv.resizeWindow("Color Filter", 800, 600)

        # create trackbars for RGB
        cv.createTrackbar("R", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("G", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("B", "Color Filter", 255, 255, nothing)

        # create trackbars for HSV
        cv.createTrackbar("Hue", "Color Filter", 179, 179, nothing)
        cv.createTrackbar("Saturation", "Color Filter", 255, 255, nothing)
        cv.createTrackbar("Value", "Color Filter", 255, 255, nothing)

        # create filter type selector
        cv.createTrackbar("Filter (0:RGB 1:HSV)", "Color Filter", 0, 1, nothing)

        while True:
            # get RGB values
            r = cv.getTrackbarPos("R", "Color Filter")
            g = cv.getTrackbarPos("G", "Color Filter")
            b = cv.getTrackbarPos("B", "Color Filter")

            # get HSV values
            h = cv.getTrackbarPos("Hue", "Color Filter")
            s = cv.getTrackbarPos("Saturation", "Color Filter")
            v = cv.getTrackbarPos("Value", "Color Filter")

            # get filter type
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
            
            # stack images horizontally with proper spacing
            display = np.hstack((image, result))
            
            # show result
            cv.imshow("Color Filter", display)

            if cv.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cv.destroyAllWindows()


    def crop_board(self, image_path, board_contour) -> np.ndarray:
        image_path = format_path(image_path)

        image = cv.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # compute the bounding box of the contour
        x, y, w, h = cv.boundingRect(board_contour)

        cropped_image = image[y:y+h, x:x+w]
        return cropped_image


    def _find_board_contour(self, image_path):
        image_path = format_path(image_path)

        image = cv.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # filter hue above 70
        lower_bound = np.array([0, 39, 149])
        upper_bound = np.array([255, 255, 255])

        mask = cv.inRange(image, lower_bound, upper_bound)

        image = cv.bitwise_and(image, image, mask=mask)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray, (7, 7), 0)

        edges = cv.Canny(blur, 60, 80, apertureSize=3)
    
        # remove noise
        kernel = np.ones((5,5), np.uint8)    
        dilated_edges = cv.dilate(edges, kernel, iterations=2)
        dilated_edges = cv.erode(dilated_edges, kernel, iterations=1)

        contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv.contourArea)
        
        # calculate the center of the contour
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2

        # create a square contour around the center
        board_size = 1470
        half_size = board_size // 2
        square_contour = np.array([
            [[center_x - half_size, center_y - half_size]],  # Top-left
            [[center_x + half_size, center_y - half_size]],  # Top-right
            [[center_x + half_size, center_y + half_size]],  # Bottom-right
            [[center_x - half_size, center_y + half_size]]   # Bottom-left
        ], dtype=np.int32)

        return square_contour
    

    def find_added_piece_coordinates(self, diff_board: np.ndarray) -> tuple:
        w, h = 105, 105
        # Define the center region size (about 60% of the tile)
        center_margin = int(w * 0.2)
        
        max_mean = 0
        max_mean_x = 0
        max_mean_y = 0

        for x in range(0, diff_board.shape[1] - w, w):
            for y in range(0, diff_board.shape[0] - h, h):
                tile = diff_board[y:y+h, x:x+w]

                # extract the center region
                center = tile[center_margin:-center_margin, center_margin:-center_margin]
                
                mean = np.mean(center)
                if mean > max_mean:
                    max_mean = mean
                    max_mean_x = x
                    max_mean_y = y

        return max_mean_x, max_mean_y
    


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
        if rect is not None:
            x1, y1, x2, y2 = rect
            return cv.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)
        return image


    def find_difference_between_images(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        if image1 is None or image2 is None:
            raise FileNotFoundError("Images not found")
        
        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        # ensure both images have the same dimensions
        gray1 = cv.resize(gray1, (gray2.shape[1], gray2.shape[0]))

        blur1 = cv.GaussianBlur(gray1, (9, 9), 0)
        blur2 = cv.GaussianBlur(gray2, (9, 9), 0)

        difference = cv.absdiff(blur1, blur2)

        # filter out noise
        _, difference = cv.threshold(difference, 40, 255, cv.THRESH_BINARY)

        # remove noise
        kernel = np.ones((9, 9), np.uint8)
        difference = cv.morphologyEx(difference, cv.MORPH_OPEN, kernel)
        difference = cv.morphologyEx(difference, cv.MORPH_CLOSE, kernel)

        return difference
    


    def find_contous_centroid(self, contours):
        moments = cv.moments(contours)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return cX, cY
    
    
    def find_contous_centroid(self, contours):
        moments = cv.moments(contours)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return cX, cY
    

    def split_board_in_blocks(self, board: np.ndarray) -> tuple:
        no_of_blocks_per_side = 14

        result = board.copy()
        
        
        # Get board dimensions

        # Get board dimensions
        height, width = board.shape[:2]

        base_block_height = height // no_of_blocks_per_side
        base_block_width = width // no_of_blocks_per_side
        
        remaining_height = height % no_of_blocks_per_side
        remaining_width = width % no_of_blocks_per_side
        
        # dividing by no_of_blocks may result in some extra pixels
        # we need to distribute these extra pixels from the outer blocks to the inner blocks
        extra_height_blocks = remaining_height // 2  # blocks from top and bottom
        extra_width_blocks = remaining_width // 2    # blocks from left and right
        
        block_heights = []
        block_widths = []
        
        for i in range(14):
            if i < extra_height_blocks or i >= 14 - extra_height_blocks:
                # Outer blocks get an extra pixel
                block_heights.append(base_block_height + 1)
            else:
                block_heights.append(base_block_height)
                
        
        for j in range(14):
            if j < extra_width_blocks or j >= 14 - extra_width_blocks:
                # Outer blocks get an extra pixel
                block_widths.append(base_block_width + 1)
            else:
                block_widths.append(base_block_width)
                
        # calculate the y and x positions of the blocks
        y_positions = [0]
        for h in block_heights:
            y_positions.append(y_positions[-1] + h)
            
        x_positions = [0]
        for w in block_widths:
            x_positions.append(x_positions[-1] + w)
        
        # draw the grid
        rectangles = []
        for i in range(14):
            for j in range(14):
                # block coordinates
                x1 = x_positions[j]
                y1 = y_positions[i]
                x2 = x_positions[j + 1]
                y2 = y_positions[i + 1]
                
                # store the rectangle coordinates
                rectangles.append(((x1, y1), (x2, y2)))
                
                # draw the rectangle
                # cv.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                
                # Optional: Add block coordinates for debugging
                # cv.putText(result, f"{i},{j}", (x1 + 5, y1 + 20),
                #           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        

                # Optional: Add block coordinates for debugging
                # cv.putText(result, f"{i},{j}", (x1 + 5, y1 + 20),
                #           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result, rectangles




if __name__ == "__main__":
    image_processing = ImageProcessing()
    i = 1
    image_processing.cv2_trackbar(f"../data/pieces/{i}.png")