class BasePipeline:
    
    def __calculate_intersection_area(self, rect1, rect2):
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0
        
        return (x_right - x_left) * (y_bottom - y_top)


    def find_matching_block(self, contour_rect, grid_rectangles) -> tuple:
        max_overlap = 0
        best_block_index = -1
        
        # convert grid rectangles to [x1, y1, x2, y2] format
        formatted_grid_rects = [[r[0][0], r[0][1], r[1][0], r[1][1]] for r in grid_rectangles]
        
        # calculate contour rectangle area
        contour_area = (contour_rect[2] - contour_rect[0]) * (contour_rect[3] - contour_rect[1])
        
        for idx, grid_rect in enumerate(formatted_grid_rects):
            intersection = self.__calculate_intersection_area(contour_rect, grid_rect)
            
            overlap_percentage = (intersection / contour_area) * 100 if contour_area > 0 else 0
            
            if overlap_percentage > max_overlap:
                max_overlap = overlap_percentage
                best_block_index = idx
        
        return best_block_index, max_overlap


    def convert_index_to_coordinates(self, index):
        divider = 14
        row = index // divider
        col = index % divider

        return f"{row+1}{chr(col+65)}"