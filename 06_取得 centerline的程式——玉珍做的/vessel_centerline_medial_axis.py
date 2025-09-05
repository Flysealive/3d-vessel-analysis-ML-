#!/usr/bin/env python3
"""
血管中軸線（Medial Axis）提取系統
使用體素化和細化算法提取真正的血管中心線
"""

import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
import os
import glob

class VesselMedialAxisExtractor:
    def __init__(self, stl_file_path):
        self.stl_file_path = stl_file_path
        self.mesh = None
        self.voxel_grid = None
        self.skeleton = None
        self.centerline = None
        self.resolution = 1.0  # 體素分辨率 (mm)
        
    def load_mesh(self):
        """載入STL網格"""
        print(f"\n載入STL檔案: {os.path.basename(self.stl_file_path)}")
        self.mesh = trimesh.load(self.stl_file_path)
        print(f"  頂點數: {len(self.mesh.vertices)}")
        print(f"  面數: {len(self.mesh.faces)}")
        return self.mesh
    
    def voxelize_mesh(self, resolution=1.0):
        """將網格體素化 - 改進版本確保更好的內部填充"""
        print("\n體素化網格...")
        
        vertices = self.mesh.vertices
        
        # 計算邊界
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        
        # 加入邊界緩衝
        buffer = 5
        min_coords -= buffer
        max_coords += buffer
        
        # 計算網格大小
        grid_size = ((max_coords - min_coords) / resolution).astype(int) + 1
        print(f"  網格大小: {grid_size}")
        
        # 創建體素網格
        voxel_grid = np.zeros(grid_size, dtype=bool)
        
        # 將頂點映射到體素
        voxel_coords = ((vertices - min_coords) / resolution).astype(int)
        
        # 填充體素
        for coord in voxel_coords:
            if all(coord >= 0) and all(coord < grid_size):
                voxel_grid[coord[0], coord[1], coord[2]] = True
        
        # 改進的填充內部方法
        print("  填充內部體素...")
        # 先較大的膨脹確保連接
        voxel_grid = ndimage.binary_dilation(voxel_grid, iterations=3)
        # 填充所有內部空洞
        voxel_grid = ndimage.binary_fill_holes(voxel_grid)
        # 較小的腐蝕以保持形狀
        voxel_grid = ndimage.binary_erosion(voxel_grid, iterations=2)
        
        self.voxel_grid = voxel_grid
        self.min_coords = min_coords
        self.resolution = resolution
        
        filled_voxels = np.sum(voxel_grid)
        print(f"  填充體素數: {filled_voxels}")
        
        return voxel_grid
    
    def extract_skeleton(self):
        """提取3D骨架（中軸線）"""
        print("\n提取3D骨架...")
        
        if self.voxel_grid is None:
            self.voxelize_mesh()
        
        # 使用3D細化算法提取骨架
        print("  執行3D細化...")
        skeleton = skeletonize_3d(self.voxel_grid)
        
        # 獲取骨架點
        skeleton_coords = np.argwhere(skeleton)
        
        if len(skeleton_coords) == 0:
            print("  警告：無法提取骨架")
            return None
        
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
        
        # 計算每個骨架點的連接數（度數）
        connectivity = np.zeros(len(self.skeleton))
        
        # 設定鄰域距離閾值
        neighbor_threshold = 5.0
        
        for i in range(len(self.skeleton)):
            # 計算到其他點的距離
            distances = np.linalg.norm(self.skeleton - self.skeleton[i], axis=1)
            # 計算鄰居數量（排除自己）
            connectivity[i] = np.sum((distances > 0) & (distances < neighbor_threshold))
        
        # 找端點：連接數最少的點
        # 端點通常只有1-2個鄰居
        endpoint_candidates_idx = np.where(connectivity <= 2)[0]
        
        if len(endpoint_candidates_idx) < 2:
            # 備用方法：使用極值點
            print("  使用極值點方法")
            # 起點：最高的Z
            start_idx = np.argmax(self.skeleton[:, 2])
            # 終點：距離起點最遠的點
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
        
        print(f"  起點: 索引{start_idx}, 位置({self.skeleton[start_idx, 0]:.1f}, "
              f"{self.skeleton[start_idx, 1]:.1f}, {self.skeleton[start_idx, 2]:.1f})")
        print(f"  終點: 索引{end_idx}, 位置({self.skeleton[end_idx, 0]:.1f}, "
              f"{self.skeleton[end_idx, 1]:.1f}, {self.skeleton[end_idx, 2]:.1f})")
        print(f"  端點間距離: {np.linalg.norm(self.skeleton[end_idx] - self.skeleton[start_idx]):.1f}mm")
        
        return start_idx, end_idx
    
    def find_shortest_path(self, start_idx, end_idx):
        """使用Dijkstra算法找從起點到終點的最短路徑"""
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
            print("  警告：無法找到完整路徑，使用備用方法")
            return self.simple_nearest_neighbor(start_idx)
        
        return path
    
    def order_skeleton_points(self):
        """排序骨架點形成連續的中心線（改進版）"""
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
        
        # 增加插值點以獲得更平滑的曲線
        self.centerline = self.interpolate_centerline(self.centerline, num_points=200)
        
        # 平滑中心線
        print("  平滑中心線...")
        for i in range(3):  # X, Y, Z座標
            self.centerline[:, i] = gaussian_filter1d(self.centerline[:, i], sigma=1.5)
        
        print(f"  中心線點數: {len(self.centerline)}")
        return self.centerline
    
    def find_path_to_target(self, start_idx, target_point):
        """找到從起點到目標終點的路徑"""
        ordered_points = []
        visited = set()
        current_idx = start_idx
        
        while len(visited) < len(self.skeleton):
            ordered_points.append(self.skeleton[current_idx])
            visited.add(current_idx)
            
            # 計算到目標的距離
            current_to_target = np.linalg.norm(self.skeleton[current_idx] - target_point)
            
            # 如果很接近目標點，結束
            if current_to_target < 5.0:
                break
            
            # 找下一個點：優先考慮朝向目標的點
            best_idx = None
            best_score = float('inf')
            
            for i in range(len(self.skeleton)):
                if i not in visited:
                    # 計算候選點的評分
                    dist_to_current = np.linalg.norm(self.skeleton[i] - self.skeleton[current_idx])
                    dist_to_target = np.linalg.norm(self.skeleton[i] - target_point)
                    
                    # 只考慮附近的點
                    if dist_to_current < 10.0:
                        # 評分：既要近又要朝向目標
                        score = dist_to_current + dist_to_target * 0.5
                        
                        if score < best_score:
                            best_score = score
                            best_idx = i
            
            if best_idx is None:
                break
            
            current_idx = best_idx
        
        return ordered_points
    
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
    
    def interpolate_centerline(self, points, num_points=200):
        """使用三次樣條插值生成平滑曲線"""
        from scipy.interpolate import interp1d
        
        # 計算累積距離作為參數
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
        
        for i in range(3):  # X, Y, Z
            f = interp1d(distances, points[:, i], kind='cubic', fill_value='extrapolate')
            interpolated.append(f(new_t))
        
        return np.column_stack(interpolated)
    
    def simplify_centerline(self, tolerance=2.0):
        """使用Douglas-Peucker算法簡化中心線但保持關鍵形狀"""
        if self.centerline is None:
            return None
        
        print(f"\n優化中心線曲線...")
        
        # 使用Douglas-Peucker算法
        def douglas_peucker(points, tolerance):
            if len(points) <= 2:
                return points
            
            # 找到離首尾連線最遠的點
            line_start = points[0]
            line_end = points[-1]
            
            max_dist = 0
            max_idx = 0
            
            for i in range(1, len(points) - 1):
                dist = self.point_to_line_distance_3d(points[i], line_start, line_end)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # 如果最大距離大於容差，遞歸處理
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
        
        t = np.dot(point_vec, line_vec) / (line_len ** 2)
        t = np.clip(t, 0, 1)
        
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)
    
    def create_visualization(self):
        """創建視覺化含功能按鈕"""
        fig = go.Figure()
        
        # 1. 血管網格（半透明）
        if self.mesh is not None:
            vertices = self.mesh.vertices
            
            # 採樣顯示
            sample_size = min(10000, len(vertices))
            sample_idx = np.random.choice(len(vertices), sample_size, replace=False)
            sample_v = vertices[sample_idx]
            
            fig.add_trace(go.Scatter3d(
                x=sample_v[:, 0],
                y=sample_v[:, 1],
                z=sample_v[:, 2],
                mode='markers',
                name='血管',
                marker=dict(
                    size=1.5,
                    color='lightblue',
                    opacity=0.2
                ),
                hoverinfo='skip'
            ))
        
        # 2. 中心線（主要顯示）
        if self.centerline is not None:
            fig.add_trace(go.Scatter3d(
                x=self.centerline[:, 0],
                y=self.centerline[:, 1],
                z=self.centerline[:, 2],
                mode='lines+markers',
                name='中心線',
                line=dict(
                    color='red',
                    width=6
                ),
                marker=dict(
                    size=3,
                    color='darkred'
                ),
                hovertemplate='中心線<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            ))
            
            # 標記起點（頂部）
            fig.add_trace(go.Scatter3d(
                x=[self.centerline[0, 0]],
                y=[self.centerline[0, 1]],
                z=[self.centerline[0, 2]],
                mode='markers+text',
                name='起點',
                marker=dict(
                    size=8,
                    color='green',
                    symbol='circle'
                ),
                text=['起點'],
                textposition='top center',
                textfont=dict(size=10)
            ))
            
            # 標記終點（底部）
            fig.add_trace(go.Scatter3d(
                x=[self.centerline[-1, 0]],
                y=[self.centerline[-1, 1]],
                z=[self.centerline[-1, 2]],
                mode='markers+text',
                name='終點',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='square'
                ),
                text=['終點'],
                textposition='top center',
                textfont=dict(size=10)
            ))
        
        # 設置佈局含功能按鈕
        fig.update_layout(
            title={
                'text': '<b>血管中心線分析</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=20)
            },
            scene=dict(
                xaxis=dict(title='X (mm)', gridcolor='lightgray'),
                yaxis=dict(title='Y (mm)', gridcolor='lightgray'),
                zaxis=dict(title='Z (mm)', gridcolor='lightgray'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            showlegend=True,
            width=1500,
            height=900,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 0.8}}],
                            label="預設視角",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera.eye": {"x": 0, "y": 2, "z": 0}}],
                            label="前視圖",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera.eye": {"x": 2, "y": 0, "z": 0}}],
                            label="側視圖",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2}}],
                            label="上視圖",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        return fig
    
    def create_centerline_stl(self, centerline_points, radius=0.75, segments=16):
        """將中心線轉換為完全連接的管狀STL網格"""
        from scipy.interpolate import splprep, splev
        import trimesh
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
    
    def save_results(self, output_dir):
        """保存結果（完全符合batch_10_vessels_organized格式）"""
        # 取得檔案名稱（只保留型號部分，如 3DModel1038604）
        base_name = os.path.basename(self.stl_file_path)[:-4]
        # 如果包含 _bottom_crop_smooth 或其他後綴，只保留型號
        if '_' in base_name:
            base_name = base_name.split('_')[0]
        
        # 創建資料夾結構（刪除figure1_original和figure3_angle_analysis，重新排序）
        dirs = {
            'interactive_3D_html': os.path.join(output_dir, 'interactive_3D_html'),
            'centerlines': os.path.join(output_dir, 'centerlines'),
            'centerlines_STL': os.path.join(output_dir, 'centerlines_STL'),  # 中心線STL資料夾
            'figure1_with_centerline': os.path.join(output_dir, 'figure1_with_centerline'),  # 3D視圖含中心線
            'figure2_xy_projection': os.path.join(output_dir, 'figure2_xy_projection'),  # XY投影
            'figure3_xz_projection': os.path.join(output_dir, 'figure3_xz_projection'),  # XZ投影
            'figure4_yz_projection': os.path.join(output_dir, 'figure4_yz_projection'),  # YZ投影
            'STL_cropped': os.path.join(output_dir, 'STL_cropped')
        }
        
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # 1. 保存互動式HTML
        fig = self.create_visualization()
        html_file = os.path.join(dirs['interactive_3D_html'], f"{base_name}.html")
        fig.write_html(html_file)
        print(f"\n✓ interactive_3D_html: {base_name}.html")
        
        # 2. 保存中心線數據
        if self.centerline is not None:
            centerline_npy = os.path.join(dirs['centerlines'], f"{base_name}.npy")
            np.save(centerline_npy, self.centerline)
            print(f"✓ centerlines: {base_name}.npy")
            
            # 生成並保存中心線STL
            centerline_mesh = self.create_centerline_stl(self.centerline, radius=1.5, segments=12)
            centerline_stl = os.path.join(dirs['centerlines_STL'], f"{base_name}.stl")
            centerline_mesh.export(centerline_stl)
            print(f"✓ centerlines_STL: {base_name}.stl")
        
        # 3. 生成各種視覺化圖像
        self.save_all_figures(dirs, base_name)
        
        # 4. 複製STL檔案
        import shutil
        stl_output = os.path.join(dirs['STL_cropped'], f"{base_name}.stl")
        shutil.copy2(self.stl_file_path, stl_output)
        print(f"✓ STL_cropped: {base_name}.stl")
        
        # 5. 生成README和SUMMARY
        self.save_documentation(output_dir, base_name)
        
        return html_file
    
    def save_all_figures(self, dirs, base_name):
        """保存所有圖像（刪除原始圖和角度分析，重新排序）"""
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
        # 起點在頂部（第一個點）
        ax.scatter(*self.centerline[0], color='green', s=100, marker='o', edgecolors='darkgreen', linewidth=2)
        # 終點在底部（最後一個點）
        ax.scatter(*self.centerline[-1], color='blue', s=100, marker='s', edgecolors='darkblue', linewidth=2)
        
        # 移除座標軸
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
        # 起點在頂部
        ax.scatter(self.centerline[0, 0], self.centerline[0, 1], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        # 終點在底部
        ax.scatter(self.centerline[-1, 0], self.centerline[-1, 1], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        # 移除座標軸
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
        # 起點在頂部（Z較高）
        ax.scatter(self.centerline[0, 0], self.centerline[0, 2], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        # 終點在底部（Z較低）
        ax.scatter(self.centerline[-1, 0], self.centerline[-1, 2], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        # 移除座標軸
        ax.set_axis_off()
        ax.grid(False)
        
        plt.savefig(os.path.join(dirs['figure3_xz_projection'], f"{base_name}.png"), 
                   dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"✓ figure3_xz_projection: {base_name}.png")
        
        # Figure 4: YZ Projection with Centerline
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.scatter(sample_v[:, 1], sample_v[:, 2], c='steelblue', s=1.0, alpha=0.6, edgecolors='none')
        ax.plot(self.centerline[:, 1], self.centerline[:, 2], 'r-', linewidth=4, alpha=0.9)
        # 起點在頂部（Z較高）
        ax.scatter(self.centerline[0, 1], self.centerline[0, 2], 
                  color='green', s=120, marker='o', edgecolors='darkgreen', linewidth=2, zorder=5)
        # 終點在底部（Z較低）
        ax.scatter(self.centerline[-1, 1], self.centerline[-1, 2], 
                  color='blue', s=120, marker='s', edgecolors='darkblue', linewidth=2, zorder=5)
        
        # 移除座標軸（與其他投影一致）
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
                f.write("- Algorithm: 3D Medial Axis Extraction\n")
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

def main():
    # 設定路徑
    stl_dir = "/Users/julie/folder/vessel/STL_smooth_solid"
    output_dir = "/Users/julie/folder/vessel_centerline_organized"  # 更改輸出資料夾名稱
    
    print("="*60)
    print("血管中軸線提取系統")
    print("="*60)
    
    # 獲取STL檔案
    stl_files = glob.glob(os.path.join(stl_dir, "*.stl"))
    stl_files.sort()  # 排序檔案
    
    if not stl_files:
        print(f"\n⚠ 在 {stl_dir} 中沒有找到STL檔案")
        return
    
    # 選擇要處理的檔案（處理所有檔案）
    test_files = stl_files  # 處理所有檔案
    
    print(f"\n將處理 {len(test_files)} 個檔案:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")
    
    extractors = []
    
    for test_file in test_files:
        print("\n" + "="*60)
        print(f"處理檔案: {os.path.basename(test_file)}")
        print("="*60)
        
        # 處理
        extractor = VesselMedialAxisExtractor(test_file)
        
        # 載入網格
        extractor.load_mesh()
        
        # 體素化
        extractor.voxelize_mesh(resolution=1.0)
        
        # 提取骨架
        extractor.extract_skeleton()
        
        # 排序形成中心線
        extractor.order_skeleton_points()
        
        # 優化中心線曲線
        extractor.simplify_centerline(tolerance=1.0)  # 優化並平滑曲線
        
        # 保存結果
        html_file = extractor.save_results(output_dir)
        
        print(f"\n✅ 完成 {os.path.basename(test_file)}！")
        
        extractors.append(extractor)
    
    print("\n" + "="*60)
    print("✅ 所有檔案處理完成！")
    print(f"📊 輸出位置: {output_dir}")
    
    return extractors

if __name__ == "__main__":
    extractor = main()