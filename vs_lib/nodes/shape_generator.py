#!/usr/bin/env python3
import numpy as np
import math

class ShapeGenerator:
    def __init__(self, safe_zone_cm=7.0):
        """
        safe_zone_cm: Kích thước vùng an toàn (mặc định 7cm theo vision_node).
        Hình vẽ sẽ được scale để nằm gọn trong bán kính này.
        """
        # Bán kính tối đa cho phép (tính từ tâm ra cạnh)
        self.max_radius = (safe_zone_cm / 100.0) / 2.0 

    def _format_stroke(self, points_list):
        """Chuyển đổi list (x,y) thành định dạng chuẩn cho Executor [x, y, 0, 1]"""
        stroke = []
        for pt in points_list:
            # Z = 0.0 vì vẽ trên mặt phẳng board
            stroke.append(np.array([pt[0], pt[1], 0.0, 1.0]))
        return [stroke] # Trả về mảng chứa 1 nét vẽ (stroke)

    def polygon(self, n_sides, scale=1.0):
        """
        Vẽ đa giác đều (Tam giác, Ngũ giác, Lục giác...)
        n_sides: Số cạnh
        scale: Tỷ lệ lấp đầy vùng safe_zone (0.0 - 1.0)
        """
        points = []
        radius = self.max_radius * scale
        
        # Xoay góc pha ban đầu để đỉnh nhọn hướng lên trên (trục Y+) hoặc theo thẩm mỹ
        # Với tam giác (3), offset -pi/2 để đáy nằm ngang, đỉnh hướng lên
        offset_angle = -math.pi / 2 
        
        for i in range(n_sides + 1): # +1 để điểm cuối trùng điểm đầu (khép kín)
            theta = offset_angle + (2 * math.pi * i / n_sides)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta) # Lưu ý: Vision trục Y hướng xuống hay lên tùy config, nhưng ở đây là đối xứng tâm
            points.append((x, y))
        return self._format_stroke(points)

    def rectangle(self, width_ratio=1.0, height_ratio=1.0):
        """
        Vẽ hình chữ nhật tâm O
        width_ratio, height_ratio: Tỷ lệ so với safe_zone (0.0 - 1.0)
        """
        w = self.max_radius * width_ratio
        h = self.max_radius * height_ratio
        
        points = [
            (-w, -h), # Góc dưới trái
            ( w, -h), # Góc dưới phải
            ( w,  h), # Góc trên phải
            (-w,  h), # Góc trên trái
            (-w, -h)  # Khép kín
        ]
        return self._format_stroke(points)

    def circle(self, resolution=36, scale=1.0):
        """Vẽ hình tròn (thực chất là đa giác nhiều cạnh)"""
        return self.polygon(n_sides=resolution, scale=scale)

    def star(self, scale=1.0):
        """Vẽ ngôi sao 5 cánh"""
        points = []
        outer_r = self.max_radius * scale
        inner_r = outer_r * 0.4 # Bán kính trong bằng 40% bán kính ngoài
        
        # Ngôi sao 5 cánh có 10 đỉnh (5 lồi, 5 lõm)
        for i in range(11): 
            angle = -math.pi/2 + (i * math.pi / 5)
            r = outer_r if i % 2 == 0 else inner_r
            points.append((r * math.cos(angle), r * math.sin(angle)))
            
        return self._format_stroke(points)

    def line(self, angle_deg=0, scale=1.0):
        """Vẽ đường thẳng qua tâm để test rung lắc"""
        length = self.max_radius * scale
        rad = math.radians(angle_deg)
        
        p1 = (-length * math.cos(rad), -length * math.sin(rad))
        p2 = ( length * math.cos(rad),  length * math.sin(rad))
        
        return self._format_stroke([p1, p2])