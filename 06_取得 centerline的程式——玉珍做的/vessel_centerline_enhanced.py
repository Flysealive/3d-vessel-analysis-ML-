#!/usr/bin/env python3
"""
血管中心線提取 - 增強版
Enhanced Vessel Centerline Extraction with Improved Centering and Endpoint Detection

整合了以下改進：
1. 端點檢測優化：使用連接度分析找出真正的血管端點
2. 路徑優化：Dijkstra最短路徑算法確保連續路徑
3. 中心性改進：距離場方法確保中心線通過血管腔中央
"""

import numpy as np
import trimesh
from skimage import measure, morphology
from scipy.ndimage import gaussian_filter1d, distance_transform_edt, gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import shutil

class EnhancedVesselExtractor:
    """增強版血管中心線提取器"""
    
    def __init__(self, stl_file_path):
        """初始化"""
        self.stl_file_path = stl_file_path
        self.mesh = None
        self.voxel_grid = None
        self.skeleton = None
        self.centerline = None
        self.resolution = 1.0
        self.min_coords = None
        self.max_coords = None
        
    def load_mesh(self):
        """載入STL網格檔案"""
        print(f"\n載入STL檔案: {os.path.basename(self.stl_file_path)}")
        self.mesh = trimesh.load(self.stl_file_path, force='mesh')
        print(f"  頂點數: {len(self.mesh.vertices)}")
        print(f"  面數: {len(self.mesh.faces)}")
        return self.mesh
    
    def voxelize_mesh(self, resolution=1.0):
        """將網格體素化"""
        if self.mesh is None:
            return None
        
        print(f"\n體素化網格...")
        self.resolution = resolution
        
        # 取得邊界
        self.min_coords = self.mesh.bounds[0]
        self.max_coords = self.mesh.bounds[1]
        
        # 計算網格大小
        grid_size = np.ceil((self.max_coords - self.min_coords) / resolution).astype(int) + 2
        print(f"  網格大小: {grid_size}")
        
        # 創建體素網格
        voxel_grid = np.zeros(grid_size, dtype=bool)
        
        # 將頂點轉換為體素座標
        voxel_coords = ((self.mesh.vertices - self.min_coords) / resolution).astype(int)
        
        # 填充體素網格
        for coord in voxel_coords:
            voxel_grid[coord[0], coord[1], coord[2]] = True
        
        # 填充內部
        print("  填充內部體素...")
        from scipy.ndimage import binary_fill_holes
        voxel_grid = binary_fill_holes(voxel_grid)
        
        self.voxel_grid = voxel_grid
        print(f"  填充體素數: {np.sum(voxel_grid)}")
        return voxel_grid
    
    def extract_skeleton_with_distance_field(self):
        """使用距離場方法提取更準確的中心線"""
        if self.voxel_grid is None:
            return None
        
        print("\n提取3D骨架（距離場增強）...")
        
        # 計算距離轉換
        distance_field = distance_transform_edt(self.voxel_grid)
        
        # 應用高斯濾波平滑距離場
        distance_field_smooth = gaussian_filter(distance_field, sigma=1.0)
        
        # 使用更嚴格的閾值來提取中心區域
        threshold = np.percentile(distance_field_smooth[distance_field_smooth > 0], 70)
        central_region = distance_field_smooth > threshold
        
        # 對中心區域進行骨架化
        print("  執行3D細化...")
        skeleton = morphology.skeletonize_3d(central_region)
        
        # 提取骨架點座標
        skeleton_coords = np.column_stack(np.where(skeleton))
        
        if len(skeleton_coords) == 0:
            # 備用方法：標準骨架化
            print("  使用標準骨架化...")
            skeleton = morphology.skeletonize_3d(self.voxel_grid)
            skeleton_coords = np.column_stack(np.where(skeleton))
        
        print(f"  骨架點數: {len(skeleton_coords)}")
        
        # 轉換回世界座標
        skeleton_world = skeleton_coords * self.resolution + self.min_coords
        
        self.skeleton = skeleton_world
        return skeleton_world
    
    def find_vessel_endpoints(self):
        """找到血管的真正端點"""
        if self.skeleton is None:
            return None, None
        
        print("\n尋找血管端點...")
        
        # 計算每個骨架點的連接數
        connectivity = np.zeros(len(self.skeleton))
        neighbor_threshold = 5.0
        
        for i in range(len(self.skeleton)):
            distances = np.linalg.norm(self.skeleton - self.skeleton[i], axis=1)
            connectivity[i] = np.sum((distances > 0) & (distances < neighbor_threshold))
        
        # 找端點：連接數最少的點
        endpoint_candidates_idx = np.where(connectivity <= 2)[0]
        
        if len(endpoint_candidates_idx) < 2:
            # 備用方法：使用極值點
            print("  使用極值點方法")
            start_idx = np.argmax(self.skeleton[:, 2])
            distances_from_start = np.linalg.norm(self.skeleton - self.skeleton[start_idx], axis=1)
            end_idx = np.argmax(distances_from_start)
        else:
            # 從候選端點中選擇
            endpoint_candidates = self.skeleton[endpoint_candidates_idx]
            
            # 起點：Z值最高的端點
            z_values = endpoint_candidates[:, 2]
            start_local_idx = np.argmax(z_values)
            start_idx = endpoint_candidates_idx[start_local_idx]
            
            # 終點：離起點最遠的端點
            distances_from_start = np.linalg.norm(
                endpoint_candidates - self.skeleton[start_idx], axis=1
            )
            end_local_idx = np.argmax(distances_from_start)
            end_idx = endpoint_candidates_idx[end_local_idx]
        
        print(f"  起點: ({self.skeleton[start_idx, 0]:.1f}, "
              f"{self.skeleton[start_idx, 1]:.1f}, {self.skeleton[start_idx, 2]:.1f})")
        print(f"  終點: ({self.skeleton[end_idx, 0]:.1f}, "
              f"{self.skeleton[end_idx, 1]:.1f}, {self.skeleton[end_idx, 2]:.1f})")
        
        return start_idx, end_idx
    
    def find_shortest_path(self, start_idx, end_idx):
        """使用Dijkstra算法找最短路徑"""
        import heapq
        
        n = len(self.skeleton)
        
        # 建立鄰接表
        neighbor_threshold = 10.0
        adjacency = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.skeleton[i] - self.skeleton[j])
                if dist < neighbor_threshold:
                    adjacency[i].append((j, dist))
                    adjacency[j].append((i, dist))
        
        # Dijkstra算法
        distances = [float('inf')] * n
        distances[start_idx] = 0
        previous = [-1] * n
        
        pq = [(0, start_idx)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end_idx:
                break
            
            for neighbor, weight in adjacency[current]:
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        # 重建路徑
        path = []
        current = end_idx
        while current != -1:
            path.append(self.skeleton[current])
            current = previous[current]
        
        path.reverse()
        
        if len(path) == 0 or not np.array_equal(path[0], self.skeleton[start_idx]):
            # 備用：最近鄰方法
            return self.simple_nearest_neighbor(start_idx)
        
        return path
    
    def simple_nearest_neighbor(self, start_idx):
        """簡單的最近鄰方法作為備用"""
        ordered_points = []
        visited = set()
        current_idx = start_idx
        
        while len(visited) < len(self.skeleton):
            ordered_points.append(self.skeleton[current_idx])
            visited.add(current_idx)
            
            min_dist = float('inf')
            next_idx = None
            
            for i in range(len(self.skeleton)):
                if i not in visited:
                    dist = np.linalg.norm(self.skeleton[i] - self.skeleton[current_idx])
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = i
            
            if next_idx is None:
                break
            
            current_idx = next_idx
        
        return ordered_points
    
    def refine_centerline_position(self):
        """細化中心線位置，確保在血管腔中心"""
        if self.centerline is None or self.voxel_grid is None:
            return
        
        print("  細化中心線位置...")
        
        # 計算距離場
        distance_field = distance_transform_edt(self.voxel_grid)
        
        refined_centerline = []
        
        for point in self.centerline:
            # 轉換到體素座標
            voxel_coord = ((point - self.min_coords) / self.resolution).astype(int)
            
            # 在局部鄰域內搜尋最大距離值
            neighborhood_size = 3
            x, y, z = voxel_coord
            
            # 確保不超出邊界
            x_min = max(0, x - neighborhood_size)
            x_max = min(self.voxel_grid.shape[0], x + neighborhood_size + 1)
            y_min = max(0, y - neighborhood_size)
            y_max = min(self.voxel_grid.shape[1], y + neighborhood_size + 1)
            z_min = max(0, z - neighborhood_size)
            z_max = min(self.voxel_grid.shape[2], z + neighborhood_size + 1)
            
            # 提取局部區域
            local_region = distance_field[x_min:x_max, y_min:y_max, z_min:z_max]
            
            if local_region.size > 0:
                # 找到局部最大值位置
                local_max_idx = np.unravel_index(np.argmax(local_region), local_region.shape)
                
                # 計算新的體素座標
                new_voxel_coord = np.array([
                    x_min + local_max_idx[0],
                    y_min + local_max_idx[1],
                    z_min + local_max_idx[2]
                ])
                
                # 轉換回世界座標
                new_point = new_voxel_coord * self.resolution + self.min_coords
                refined_centerline.append(new_point)
            else:
                refined_centerline.append(point)
        
        # 更新中心線
        self.centerline = np.array(refined_centerline)
    
    def order_skeleton_points(self):
        """排序骨架點形成連續的中心線"""
        if self.skeleton is None:
            return None
        
        print("\n排序骨架點...")
        
        # 找到真正的端點
        start_idx, end_idx = self.find_vessel_endpoints()
        
        # 使用Dijkstra算法找最短路徑
        ordered_points = self.find_shortest_path(start_idx, end_idx)
        
        if len(ordered_points) < 10:
            print("  警告：路徑太短，使用備用方法")
            ordered_points = self.simple_nearest_neighbor(start_idx)
        
        # 轉換為numpy數組
        self.centerline = np.array(ordered_points)
        
        # 增加插值點
        self.centerline = self.interpolate_centerline(self.centerline, num_points=200)
        
        # 細化中心線位置
        self.refine_centerline_position()
        
        # 平滑中心線
        print("  平滑中心線...")
        for i in range(3):
            self.centerline[:, i] = gaussian_filter1d(self.centerline[:, i], sigma=1.0)
        
        print(f"  中心線點數: {len(self.centerline)}")
        return self.centerline
    
    def interpolate_centerline(self, points, num_points=200):
        """使用三次樣條插值生成平滑曲線"""
        # 計算累積距離
        distances = [0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            distances.append(distances[-1] + dist)
        distances = np.array(distances)
        
        # 正規化到[0, 1]
        distances = distances / distances[-1]
        
        # 對每個座標軸進行插值
        new_t = np.linspace(0, 1, num_points)
        interpolated = []
        
        for i in range(3):
            f = interp1d(distances, points[:, i], kind='cubic', fill_value='extrapolate')
            interpolated.append(f(new_t))
        
        return np.column_stack(interpolated)
    
    def simplify_centerline(self, tolerance=2.0):
        """使用Douglas-Peucker算法簡化中心線"""
        if self.centerline is None:
            return None
        
        print("\n優化中心線曲線...")
        
        def douglas_peucker(points, tolerance):
            if len(points) <= 2:
                return points
            
            line_start = points[0]
            line_end = points[-1]
            
            max_dist = 0
            max_idx = 0
            
            for i in range(1, len(points) - 1):
                dist = self.point_to_line_distance_3d(points[i], line_start, line_end)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            if max_dist > tolerance:
                left = douglas_peucker(points[:max_idx + 1], tolerance)
                right = douglas_peucker(points[max_idx:], tolerance)
                return np.vstack([left[:-1], right])
            else:
                return np.array([points[0], points[-1]])
        
        # 保持更多細節
        simplified = douglas_peucker(self.centerline, tolerance * 0.3)
        
        # 再次插值以獲得平滑曲線
        if len(simplified) > 3:
            self.centerline = self.interpolate_centerline(simplified, num_points=150)
        
        print(f"  優化後點數: {len(self.centerline)}")
        return self.centerline
    
    def point_to_line_distance_3d(self, point, line_start, line_end):
        """計算3D空間中點到線段的距離"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        
        t = np.dot(line_unitvec, point_vec_scaled)
        t = max(0.0, min(1.0, t))
        
        nearest = line_vec * t
        dist = np.linalg.norm(point_vec - nearest)
        
        return dist
    
    def create_centerline_stl(self, centerline_points, radius=0.75, segments=16):
        """將中心線轉換為完全連接的管狀STL網格"""
        from scipy.interpolate import splprep, splev
        import math
        
        # 使用樣條插值創建非常平滑的曲線
        try:
            # 增加平滑參數s和插值點數量
            tck, u = splprep([centerline_points[:, 0], 
                            centerline_points[:, 1], 
                            centerline_points[:, 2]], s=5, k=3)
            
            # 生成密集的插值點以獲得平滑曲線
            u_fine = np.linspace(0, 1, len(centerline_points) * 10)
            smooth_points = np.array(splev(u_fine, tck)).T
        except:
            # 如果樣條插值失敗，使用線性插值
            from scipy.interpolate import interp1d
            t = np.linspace(0, 1, len(centerline_points))
            t_fine = np.linspace(0, 1, len(centerline_points) * 10)
            
            fx = interp1d(t, centerline_points[:, 0], kind='linear')
            fy = interp1d(t, centerline_points[:, 1], kind='linear')
            fz = interp1d(t, centerline_points[:, 2], kind='linear')
            
            smooth_points = np.column_stack([fx(t_fine), fy(t_fine), fz(t_fine)])
        
        vertices = []
        faces = []
        
        # 計算每個點的Frenet框架
        for i, point in enumerate(smooth_points):
            # 計算切線方向
            if i == 0:
                tangent = smooth_points[1] - smooth_points[0]
            elif i == len(smooth_points) - 1:
                tangent = smooth_points[i] - smooth_points[i-1]
            else:
                tangent = smooth_points[i+1] - smooth_points[i-1]
            
            tangent = tangent / np.linalg.norm(tangent)
            
            # 找到垂直於切線的兩個向量
            if abs(tangent[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            
            binormal = np.cross(tangent, ref)
            binormal = binormal / np.linalg.norm(binormal)
            normal = np.cross(binormal, tangent)
            normal = normal / np.linalg.norm(normal)
            
            # 創建圓形截面的頂點
            for j in range(segments):
                angle = 2 * math.pi * j / segments
                offset = normal * math.cos(angle) * radius + binormal * math.sin(angle) * radius
                vertices.append(point + offset)
        
        vertices = np.array(vertices)
        
        # 創建管狀網格的面 - 確保正確連接
        for i in range(len(smooth_points) - 1):
            for j in range(segments):
                curr_base = i * segments
                next_base = (i + 1) * segments
                
                v1 = curr_base + j
                v2 = curr_base + ((j + 1) % segments)
                v3 = next_base + j
                v4 = next_base + ((j + 1) % segments)
                
                # 創建兩個三角形連接圓環
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])
        
        # 封閉兩端 - 創建扇形
        # 起始端
        start_center = len(vertices)
        vertices = np.vstack([vertices, smooth_points[0]])
        for j in range(segments):
            v1 = j
            v2 = (j + 1) % segments
            faces.append([start_center, v1, v2])
        
        # 結束端  
        end_center = len(vertices)
        vertices = np.vstack([vertices, smooth_points[-1]])
        last_base = (len(smooth_points) - 1) * segments
        for j in range(segments):
            v1 = last_base + j
            v2 = last_base + ((j + 1) % segments)
            faces.append([end_center, v2, v1])
        
        faces = np.array(faces)
        
        # 創建Trimesh對象
        tube_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 清理和修復網格
        tube_mesh.update_faces(tube_mesh.unique_faces())  # 移除重複面
        tube_mesh.update_faces(tube_mesh.nondegenerate_faces())  # 移除退化面
        tube_mesh.fix_normals()  # 修復法線方向
        
        return tube_mesh
    
    def save_interactive_html(self, output_path, base_name):
        """生成互動式3D HTML視覺化"""
        if self.centerline is None:
            return None
        
        # 準備血管網格數據（採樣）
        vertices = self.mesh.vertices
        sample_size = min(10000, len(vertices))
        sample_idx = np.random.choice(len(vertices), sample_size, replace=False)
        sample_vertices = vertices[sample_idx]
        
        # 創建血管散點
        vessel_trace = go.Scatter3d(
            x=sample_vertices[:, 0],
            y=sample_vertices[:, 1],
            z=sample_vertices[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightblue', opacity=0.3),
            name='血管'
        )
        
        # 創建中心線
        centerline_trace = go.Scatter3d(
            x=self.centerline[:, 0],
            y=self.centerline[:, 1],
            z=self.centerline[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=2, color='red'),
            name='中心線'
        )
        
        # 標記起點和終點
        start_point = go.Scatter3d(
            x=[self.centerline[0, 0]],
            y=[self.centerline[0, 1]],
            z=[self.centerline[0, 2]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle'),
            name='起點'
        )
        
        end_point = go.Scatter3d(
            x=[self.centerline[-1, 0]],
            y=[self.centerline[-1, 1]],
            z=[self.centerline[-1, 2]],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='square'),
            name='終點'
        )
        
        # 創建圖表
        fig = go.Figure(data=[vessel_trace, centerline_trace, start_point, end_point])
        
        # 設置布局
        fig.update_layout(
            title=f"{base_name} - 血管中心線視覺化（增強版）",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            width=1200,
            height=900
        )
        
        # 保存HTML
        fig.write_html(output_path)
        
        return output_path
    
    def save_results(self, output_dir):
        """保存所有結果（與方法1統一格式）"""
        # 取得檔案名稱（只保留型號部分，如 3DModel1038604）
        base_name = os.path.basename(self.stl_file_path)[:-4]
        # 如果包含 _bottom_crop_smooth 或其他後綴，只保留型號
        if '_' in base_name:
            base_name = base_name.split('_')[0]
        
        # 創建輸出目錄（與方法1相同的結構）
        dirs = {
            'interactive_3D_html': os.path.join(output_dir, 'interactive_3D_html'),
            'centerlines': os.path.join(output_dir, 'centerlines'),
            'centerlines_STL': os.path.join(output_dir, 'centerlines_STL'),
            'figure1_with_centerline': os.path.join(output_dir, 'figure1_with_centerline'),
            'figure2_xy_projection': os.path.join(output_dir, 'figure2_xy_projection'),
            'figure3_xz_projection': os.path.join(output_dir, 'figure3_xz_projection'),
            'figure4_yz_projection': os.path.join(output_dir, 'figure4_yz_projection'),
            'STL_cropped': os.path.join(output_dir, 'STL_cropped')
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. 保存互動式HTML
        html_file = os.path.join(dirs['interactive_3D_html'], f"{base_name}.html")
        self.save_interactive_html(html_file, base_name)
        print(f"\n✓ interactive_3D_html: {base_name}.html")
        
        # 2. 保存中心線數據
        if self.centerline is not None:
            np.save(os.path.join(dirs['centerlines'], f"{base_name}.npy"), self.centerline)
            print(f"✓ centerlines: {base_name}.npy")
        
        # 3. 生成並保存中心線STL
        if self.centerline is not None:
            tube_mesh = self.create_centerline_stl(self.centerline)
            tube_mesh.export(os.path.join(dirs['centerlines_STL'], f"{base_name}.stl"))
            print(f"✓ centerlines_STL: {base_name}.stl")
        
        # 4. 生成所有圖像（與方法1相同）
        self.save_all_figures(dirs, base_name)
        
        # 5. 複製原始STL
        stl_output = os.path.join(dirs['STL_cropped'], f"{base_name}.stl")
        shutil.copy2(self.stl_file_path, stl_output)
        print(f"✓ STL_cropped: {base_name}.stl")
        
        # 6. 生成文檔
        self.save_documentation(output_dir, base_name)
        
        return html_file
    
    def save_all_figures(self, dirs, base_name):
        """保存所有圖像（與方法1統一格式）"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if self.centerline is None:
            return
        
        vertices = self.mesh.vertices
        
        # 使用更多點並提高可見度
        sample_size = min(10000, len(vertices))
        sample_idx = np.random.choice(len(vertices), sample_size, replace=False)
        sample_v = vertices[sample_idx]
        
        # Figure 1: With Centerline (3D視圖含中心線)
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.scatter(sample_v[:, 0], sample_v[:, 1], sample_v[:, 2], 
                  c='steelblue', s=1.0, alpha=0.5, edgecolors='none')
        ax.plot(self.centerline[:, 0], self.centerline[:, 1], self.centerline[:, 2], 
                'r-', linewidth=4, alpha=0.9)
        ax.scatter(*self.centerline[0], color='green', s=100, marker='o', edgecolors='darkgreen', linewidth=2)
        ax.scatter(*self.centerline[-1], color='blue', s=100, marker='s', edgecolors='darkblue', linewidth=2)
        
        ax.set_axis_off()
        ax.grid(False)
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(os.path.join(dirs['figure1_with_centerline'], f"{base_name}.png"), 
                   dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ figure1_with_centerline: {base_name}.png")
        
        # Figure 2: XY Projection
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.scatter(sample_v[:, 0], sample_v[:, 1], c='steelblue', s=1.0, alpha=0.6, edgecolors='none')
        ax.plot(self.centerline[:, 0], self.centerline[:, 1], 'r-', linewidth=4, alpha=0.9)
        ax.scatter(self.centerline[0, 0], self.centerline[0, 1], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        ax.scatter(self.centerline[-1, 0], self.centerline[-1, 1], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        ax.set_axis_off()
        ax.grid(False)
        ax.set_aspect('equal')
        
        plt.savefig(os.path.join(dirs['figure2_xy_projection'], f"{base_name}.png"), 
                   dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"✓ figure2_xy_projection: {base_name}.png")
        
        # Figure 3: XZ Projection
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.scatter(sample_v[:, 0], sample_v[:, 2], c='steelblue', s=1.0, alpha=0.6, edgecolors='none')
        ax.plot(self.centerline[:, 0], self.centerline[:, 2], 'r-', linewidth=4, alpha=0.9)
        ax.scatter(self.centerline[0, 0], self.centerline[0, 2], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        ax.scatter(self.centerline[-1, 0], self.centerline[-1, 2], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        ax.set_axis_off()
        ax.grid(False)
        
        plt.savefig(os.path.join(dirs['figure3_xz_projection'], f"{base_name}.png"), 
                   dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"✓ figure3_xz_projection: {base_name}.png")
        
        # Figure 4: YZ Projection
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.scatter(sample_v[:, 1], sample_v[:, 2], c='steelblue', s=1.0, alpha=0.6, edgecolors='none')
        ax.plot(self.centerline[:, 1], self.centerline[:, 2], 'r-', linewidth=4, alpha=0.9)
        ax.scatter(self.centerline[0, 1], self.centerline[0, 2], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        ax.scatter(self.centerline[-1, 1], self.centerline[-1, 2], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        ax.set_axis_off()
        ax.grid(False)
        ax.set_aspect('equal')
        
        plt.savefig(os.path.join(dirs['figure4_yz_projection'], f"{base_name}.png"), 
                   dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"✓ figure4_yz_projection: {base_name}.png")
    
    def save_documentation(self, output_dir, base_name):
        """保存README和SUMMARY文件"""
        # README.md
        readme_file = os.path.join(output_dir, 'README.md')
        if not os.path.exists(readme_file):
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write("# Vessel Centerline Analysis Results\n\n")
                f.write("This folder contains the analysis results of vessel centerlines.\n\n")
                f.write("## Folder Structure:\n")
                f.write("- `centerlines/`: Centerline data in NumPy format\n")
                f.write("- `centerlines_STL/`: Centerline STL models\n")
                f.write("- `figure1_with_centerline/`: 3D vessel with centerline\n")
                f.write("- `figure2_xy_projection/`: XY projection view\n")
                f.write("- `figure3_xz_projection/`: XZ projection view\n")
                f.write("- `figure4_yz_projection/`: YZ projection view\n")
                f.write("- `interactive_3D_html/`: Interactive 3D visualization\n")
                f.write("- `STL_cropped/`: Processed STL files\n\n")
                f.write("## Processing Details:\n")
                f.write("- Algorithm: Enhanced Distance Field Method\n")
                f.write("- Smoothing: Cubic Spline Interpolation\n")
                f.write("- Optimization: Douglas-Peucker Simplification\n")
            print(f"✓ README.md")
        
        # SUMMARY.md
        summary_file = os.path.join(output_dir, 'SUMMARY.md')
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## {base_name}\n")
            f.write(f"- Processing Date: {np.datetime64('now')}\n")
            f.write(f"- Vertices: {len(self.mesh.vertices)}\n")
            f.write(f"- Faces: {len(self.mesh.faces)}\n")
            if self.centerline is not None:
                f.write(f"- Centerline Points: {len(self.centerline)}\n")
                
                # 計算長度
                total_length = 0
                for i in range(1, len(self.centerline)):
                    total_length += np.linalg.norm(self.centerline[i] - self.centerline[i-1])
                f.write(f"- Total Length: {total_length:.1f} mm\n")
        print(f"✓ SUMMARY.md")


# 兼容性別名
VesselMedialAxisExtractor = EnhancedVesselExtractor

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python vessel_centerline_enhanced.py <STL檔案路徑>")
        sys.exit(1)
    
    stl_file = sys.argv[1]
    output_dir = "output_enhanced"
    
    # 處理血管
    extractor = EnhancedVesselExtractor(stl_file)
    extractor.load_mesh()
    extractor.voxelize_mesh(resolution=0.8)  # 使用更高解析度
    extractor.extract_skeleton_with_distance_field()
    extractor.order_skeleton_points()
    extractor.simplify_centerline()
    extractor.save_results(output_dir)
    
    print(f"\n處理完成！結果保存在: {output_dir}")