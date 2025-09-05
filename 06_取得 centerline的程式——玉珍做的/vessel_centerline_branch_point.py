#!/usr/bin/env python3
"""
血管中軸線提取系統 - 分支點方法 (Branch Point Method)
============================================

第二個額外的中心線提取方法

特點：
1. 自動檢測血管分支點（分叉位置）
2. 選擇Z軸第二高的分支點作為起點
3. 從分支點開始提取主要中心線
4. 生成3D互動視覺化

輸出：
- CSV檔案：包含x,y,z座標的中心線點
- HTML檔案：3D互動視覺化

視覺化標記：
- 綠色圓形：起點
- 橙色正方形：終點
- 紫色菱形：所有分支點
- 黃色大菱形：選中的第二高分支點

作者：Claude Assistant
日期：2024-08-29
"""

import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
import networkx as nx
import os
import shutil

class VesselCenterlineExtractor:
    def __init__(self, stl_file_path):
        self.stl_file_path = stl_file_path
        self.mesh = None
        self.voxel_grid = None
        self.skeleton = None
        self.skeleton_points = None
        self.centerline = None
        self.resolution = 1.0
        self.endpoints = []
        self.branch_points = []
        self.min_coords = None
        
    def load_mesh(self):
        """載入STL網格"""
        print(f"\n處理檔案: {os.path.basename(self.stl_file_path)}")
        self.mesh = trimesh.load(self.stl_file_path)
        print(f"  頂點數: {len(self.mesh.vertices)}")
        print(f"  面數: {len(self.mesh.faces)}")
        return self.mesh
    
    def voxelize_mesh(self, resolution=1.0):
        """將網格體素化"""
        print("體素化網格...")
        
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
        
        # 填充內部
        print("  填充內部...")
        voxel_grid = ndimage.binary_dilation(voxel_grid, iterations=3)
        voxel_grid = ndimage.binary_fill_holes(voxel_grid)
        voxel_grid = ndimage.binary_erosion(voxel_grid, iterations=2)
        
        self.voxel_grid = voxel_grid
        self.min_coords = min_coords
        self.resolution = resolution
        
        return voxel_grid
    
    def extract_skeleton(self):
        """提取3D骨架"""
        print("提取骨架...")
        
        if self.voxel_grid is None:
            self.voxelize_mesh()
        
        # 使用3D細化算法
        skeleton = skeletonize_3d(self.voxel_grid)
        
        # 獲取骨架點
        skeleton_coords = np.argwhere(skeleton)
        
        if len(skeleton_coords) == 0:
            print("  警告：無法提取骨架")
            return None
        
        print(f"  骨架點數: {len(skeleton_coords)}")
        
        # 轉換回世界坐標
        skeleton_points = skeleton_coords * self.resolution + self.min_coords
        
        self.skeleton = skeleton
        self.skeleton_points = skeleton_points
        
        return skeleton_points
    
    def find_endpoints(self):
        """自動檢測骨架端點"""
        print("檢測端點...")
        
        if self.skeleton is None:
            return []
        
        # 計算每個骨架點的鄰居數
        endpoints = []
        skeleton_coords = np.argwhere(self.skeleton)
        
        for i, coord in enumerate(skeleton_coords):
            # 計算26鄰域中的骨架點數
            x, y, z = coord
            neighbors = 0
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        if (0 <= nx < self.skeleton.shape[0] and
                            0 <= ny < self.skeleton.shape[1] and
                            0 <= nz < self.skeleton.shape[2]):
                            
                            if self.skeleton[nx, ny, nz]:
                                neighbors += 1
            
            # 端點只有一個鄰居
            if neighbors <= 1:
                world_coord = coord * self.resolution + self.min_coords
                endpoints.append(world_coord)
        
        print(f"  找到 {len(endpoints)} 個端點")
        self.endpoints = endpoints
        
        return endpoints
    
    def find_branch_points(self):
        """檢測骨架分支點（鄰居數>=3的點）"""
        print("檢測分支點...")
        
        if self.skeleton is None:
            return []
        
        branch_points = []
        skeleton_coords = np.argwhere(self.skeleton)
        
        for i, coord in enumerate(skeleton_coords):
            x, y, z = coord
            
            # 計算26鄰域中的骨架點數
            neighbors = 0
            neighbor_coords = []
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        if (0 <= nx < self.skeleton.shape[0] and
                            0 <= ny < self.skeleton.shape[1] and
                            0 <= nz < self.skeleton.shape[2]):
                            
                            if self.skeleton[nx, ny, nz]:
                                neighbors += 1
                                neighbor_coords.append([nx, ny, nz])
            
            # 分支點有3個或更多鄰居（表示有分叉）
            if neighbors >= 3:
                # 檢查是否真的是分支點（不只是彎曲）
                # 通過檢查鄰居是否在不同方向
                if len(neighbor_coords) >= 3:
                    neighbor_array = np.array(neighbor_coords)
                    # 計算鄰居相對於當前點的方向向量
                    directions = neighbor_array - coord
                    
                    # 如果方向向量的角度分散，則是真的分支點
                    # 簡單檢查：至少有一對鄰居的夾角大於90度
                    is_branch = False
                    for i in range(len(directions)):
                        for j in range(i+1, len(directions)):
                            # 計算點積
                            dot_product = np.dot(directions[i], directions[j])
                            # 如果點積小於0，夾角大於90度
                            if dot_product < 0:
                                is_branch = True
                                break
                        if is_branch:
                            break
                    
                    if is_branch:
                        world_coord = coord * self.resolution + self.min_coords
                        branch_points.append(world_coord)
        
        print(f"  找到 {len(branch_points)} 個分支點")
        self.branch_points = branch_points
        
        return branch_points
    
    def build_skeleton_graph(self):
        """建立骨架圖結構"""
        print("建立骨架圖...")
        
        if self.skeleton_points is None:
            return None
        
        # 創建圖
        G = nx.Graph()
        
        # 添加節點
        for i, point in enumerate(self.skeleton_points):
            G.add_node(i, pos=point)
        
        # 建立KD樹找鄰居
        tree = KDTree(self.skeleton_points)
        
        # 添加邊（連接相鄰的骨架點）
        for i, point in enumerate(self.skeleton_points):
            # 找最近的6個鄰居
            distances, indices = tree.query(point, k=7)
            
            for j in range(1, len(indices)):
                if distances[j] < 2.0 * self.resolution:  # 只連接相鄰的點
                    G.add_edge(i, indices[j], weight=distances[j])
        
        return G
    
    def extract_main_centerline(self):
        """提取主要中心線（從Z軸第二高的分支點開始）"""
        print("提取主要中心線...")
        
        if self.skeleton_points is None:
            self.extract_skeleton()
        
        if len(self.endpoints) == 0:
            self.find_endpoints()
        
        # 檢測分支點
        if len(self.branch_points) == 0:
            self.find_branch_points()
        
        # 建立圖
        G = self.build_skeleton_graph()
        
        # 如果有分支點，選擇Z軸第二高的分支點作為起點
        if len(self.branch_points) > 0:
            print(f"  使用分支點作為起點候選")
            
            # 對分支點按Z軸排序
            branch_points_array = np.array(self.branch_points)
            z_values_branch = branch_points_array[:, 2]
            
            # 找出不同的Z層級
            unique_z_branch = np.unique(z_values_branch)
            unique_z_branch = np.sort(unique_z_branch)[::-1]  # 降序排列
            
            print(f"  分支點Z軸範圍: {z_values_branch.min():.1f} 到 {z_values_branch.max():.1f}")
            print(f"  分支點Z層級數: {len(unique_z_branch)}")
            
            # 選擇第二高的分支點Z層級
            if len(unique_z_branch) > 1:
                second_highest_z = unique_z_branch[1]
                print(f"  選擇第二高分支點Z層級: {second_highest_z:.1f}")
            else:
                second_highest_z = unique_z_branch[0]
                print(f"  只有一個分支點層級，使用: {second_highest_z:.1f}")
            
            # 找出在這個Z層級的分支點
            branch_at_second_z = branch_points_array[np.abs(z_values_branch - second_highest_z) < 0.5]
            
            if len(branch_at_second_z) > 0:
                # 選擇中間的分支點
                start_point = branch_at_second_z[len(branch_at_second_z) // 2]
                print(f"  選擇分支點起點: ({start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f})")
            else:
                # 如果沒有找到，使用最接近的分支點
                idx = np.argmin(np.abs(z_values_branch - second_highest_z))
                start_point = branch_points_array[idx]
                print(f"  使用最接近的分支點: ({start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f})")
            
            # 找到最接近的骨架點索引
            tree = KDTree(self.skeleton_points)
            _, start_idx = tree.query(start_point)
            
        else:
            # 如果沒有分支點，回退到原來的方法
            print("  警告：未找到分支點，使用Z軸第二高的骨架點")
            
            z_values = self.skeleton_points[:, 2]
            unique_z_values = np.unique(z_values)
            unique_z_values = np.sort(unique_z_values)[::-1]
            
            if len(unique_z_values) > 1:
                second_highest_z = unique_z_values[1]
            else:
                second_highest_z = unique_z_values[0]
            
            points_at_second_z = self.skeleton_points[z_values == second_highest_z]
            
            if len(points_at_second_z) > 0:
                center_idx = len(points_at_second_z) // 2
                start_point = points_at_second_z[center_idx]
                for i, p in enumerate(self.skeleton_points):
                    if np.allclose(p, start_point):
                        start_idx = i
                        break
            else:
                z_sorted_indices = np.argsort(z_values)[::-1]
                start_idx = z_sorted_indices[1] if len(z_sorted_indices) > 1 else z_sorted_indices[0]
                start_point = self.skeleton_points[start_idx]
            
            print(f"  選擇起點: ({start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f})")
        
        # 找到離起點最遠的點作為終點
        max_length = 0
        best_path = None
        
        # 計算從起點到所有其他點的最短路徑
        try:
            lengths = nx.single_source_dijkstra_path_length(G, start_idx)
            # 找最遠的點
            farthest_idx = max(lengths, key=lengths.get)
            
            # 獲取路徑
            path = nx.shortest_path(G, source=start_idx, target=farthest_idx, weight='weight')
            best_path = path
            print(f"  找到路徑: 從點 {start_idx} 到點 {farthest_idx}, 長度 {len(path)}")
            
        except nx.NetworkXNoPath:
            print("  警告：無法找到有效路徑，嘗試備選方案")
            # 如果無法找到路徑，嘗試找最近的可達點
            for target_idx in range(len(self.skeleton_points)):
                if target_idx == start_idx:
                    continue
                try:
                    path = nx.shortest_path(G, source=start_idx, target=target_idx, weight='weight')
                    if len(path) > max_length:
                        max_length = len(path)
                        best_path = path
                except nx.NetworkXNoPath:
                    continue
        
        if best_path is None:
            print("  使用所有骨架點")
            return self.skeleton_points
        
        # 提取路徑上的點
        centerline_points = np.array([self.skeleton_points[i] for i in best_path])
        
        # 平滑中心線
        if len(centerline_points) > 5:
            centerline_points = self.smooth_centerline(centerline_points)
        
        self.centerline = centerline_points
        print(f"  中心線點數: {len(centerline_points)}")
        
        return centerline_points
    
    def smooth_centerline(self, points, sigma=2):
        """平滑中心線"""
        smoothed = np.copy(points)
        for i in range(3):  # 對每個坐標維度
            smoothed[:, i] = gaussian_filter1d(points[:, i], sigma=sigma, mode='nearest')
        return smoothed
    
    def create_interactive_plot(self):
        """創建3D互動視覺化"""
        fig = go.Figure()
        
        # 添加血管表面（降採樣）
        if self.mesh:
            sample_rate = max(1, len(self.mesh.vertices) // 10000)
            vertices = self.mesh.vertices[::sample_rate]
            
            fig.add_trace(go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color='lightblue',
                    opacity=0.2
                ),
                name='血管表面',
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
        
        # 添加骨架點（降採樣顯示）
        if self.skeleton_points is not None and len(self.skeleton_points) > 0:
            sample_rate = max(1, len(self.skeleton_points) // 2000)
            skeleton_sample = self.skeleton_points[::sample_rate]
            
            fig.add_trace(go.Scatter3d(
                x=skeleton_sample[:, 0],
                y=skeleton_sample[:, 1],
                z=skeleton_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='gray',
                    opacity=0.3
                ),
                name='骨架',
                hovertemplate='骨架<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
        
        # 添加中心線
        if self.centerline is not None and len(self.centerline) > 0:
            fig.add_trace(go.Scatter3d(
                x=self.centerline[:, 0],
                y=self.centerline[:, 1],
                z=self.centerline[:, 2],
                mode='lines+markers',
                line=dict(
                    color='red',
                    width=6
                ),
                marker=dict(
                    size=3,
                    color='darkred'
                ),
                name='中心線',
                hovertemplate='中心線<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
            
            # 標記起點（綠色圓形）
            fig.add_trace(go.Scatter3d(
                x=[self.centerline[0, 0]],
                y=[self.centerline[0, 1]],
                z=[self.centerline[0, 2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    symbol='circle'
                ),
                name='起點',
                hovertemplate='起點<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
            
            # 標記終點（橙色正方形）
            fig.add_trace(go.Scatter3d(
                x=[self.centerline[-1, 0]],
                y=[self.centerline[-1, 1]],
                z=[self.centerline[-1, 2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='orange',
                    symbol='square'
                ),
                name='終點',
                hovertemplate='終點<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
        
        # 添加檢測到的端點（已註釋，因為現在忽略其他端點）
        # if len(self.endpoints) > 0:
        #     endpoints_array = np.array(self.endpoints)
        #     fig.add_trace(go.Scatter3d(
        #         x=endpoints_array[:, 0],
        #         y=endpoints_array[:, 1],
        #         z=endpoints_array[:, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=8,
        #             color='purple',
        #             symbol='diamond'
        #         ),
        #         name='檢測端點',
        #         hovertemplate='端點<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
        #     ))
        
        # 顯示所有分支點
        if len(self.branch_points) > 0:
            branch_points_array = np.array(self.branch_points)
            fig.add_trace(go.Scatter3d(
                x=branch_points_array[:, 0],
                y=branch_points_array[:, 1],
                z=branch_points_array[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color='purple',
                    symbol='diamond'
                ),
                name='分支點',
                hovertemplate='分支點<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
            ))
            
            # 特別標記Z軸第二高的分支點
            z_values_branch = branch_points_array[:, 2]
            unique_z_branch = np.unique(z_values_branch)
            unique_z_branch = np.sort(unique_z_branch)[::-1]
            
            if len(unique_z_branch) > 1:
                second_highest_z = unique_z_branch[1]
                branch_at_second = branch_points_array[np.abs(z_values_branch - second_highest_z) < 0.5]
                
                if len(branch_at_second) > 0:
                    selected_branch = branch_at_second[len(branch_at_second) // 2]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[selected_branch[0]],
                        y=[selected_branch[1]],
                        z=[selected_branch[2]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='yellow',
                            symbol='diamond'
                        ),
                        name='第二高分支點(起點)',
                        hovertemplate='第二高分支點<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}'
                    ))
        
        # 設置佈局
        fig.update_layout(
            title=f'血管中心線 - {os.path.basename(self.stl_file_path)}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            showlegend=True,
            width=1200,
            height=900,
            hovermode='closest'
        )
        
        return fig
    
    def create_centerline_stl(self, centerline_points, radius=0.75, segments=16):
        """將中心線轉換為完全連接的管狀STL網格"""
        from scipy.interpolate import splprep, splev
        import trimesh
        import math
        import numpy as np
        
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
        fig = self.create_interactive_plot()
        html_file = os.path.join(dirs['interactive_3D_html'], f"{base_name}.html")
        fig.write_html(html_file)
        print(f"\n✓ interactive_3D_html: {base_name}.html")
        
        # 2. 保存中心線數據
        if self.centerline is not None:
            np.save(os.path.join(dirs['centerlines'], f"{base_name}.npy"), self.centerline)
            print(f"✓ centerlines: {base_name}.npy")
            
            # 生成並保存中心線STL
            centerline_mesh = self.create_centerline_stl(self.centerline, radius=1.5, segments=12)
            if centerline_mesh:
                centerline_mesh.export(os.path.join(dirs['centerlines_STL'], f"{base_name}.stl"))
                print(f"✓ centerlines_STL: {base_name}.stl")
        
        # 3. 生成所有圖像（與方法1相同）
        self.save_all_figures(dirs, base_name)
        
        # 4. 複製原始STL
        stl_output = os.path.join(dirs['STL_cropped'], f"{base_name}.stl")
        shutil.copy2(self.stl_file_path, stl_output)
        print(f"✓ STL_cropped: {base_name}.stl")
        
        # 5. 生成文檔
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
        
        # 顯示分支點
        if len(self.branch_points) > 0:
            branch_array = np.array(self.branch_points)
            ax.scatter(branch_array[:, 0], branch_array[:, 1], branch_array[:, 2],
                      color='purple', s=50, marker='D', alpha=0.8, label='Branch Points')
        
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
                f.write("- Algorithm: Branch Point Detection Method\n")
                f.write("- Feature: Automatic branch point detection\n")
                f.write("- Starting point: Second highest branch point (Z-axis)\n")
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
                f.write(f"- Branch Points Detected: {len(self.branch_points)}\n")
                
                # 計算長度
                total_length = 0
                for i in range(1, len(self.centerline)):
                    total_length += np.linalg.norm(self.centerline[i] - self.centerline[i-1])
                f.write(f"- Total Length: {total_length:.1f} mm\n")
        print(f"✓ SUMMARY.md")

def process_vessels(input_dir, output_dir):
    """處理指定的血管檔案"""
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 指定要處理的檔案
    vessel_files = [
        "3DModel822993_bottom_crop_smooth.stl",
        "3DModel872531_bottom_crop_smooth.stl"
    ]
    
    for filename in vessel_files:
        stl_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(stl_path):
            print(f"檔案不存在: {stl_path}")
            continue
        
        # 創建提取器
        extractor = VesselCenterlineExtractor(stl_path)
        
        # 載入網格
        extractor.load_mesh()
        
        # 提取中心線
        centerline = extractor.extract_main_centerline()
        
        # 創建互動視覺化
        fig = extractor.create_interactive_plot()
        
        # 儲存HTML
        html_file = os.path.join(output_dir, filename.replace('.stl', '_auto_centerline.html'))
        fig.write_html(html_file)
        print(f"  儲存互動視覺化: {html_file}")
        
        # 儲存中心線數據（格式：x,y,z，包含標題行）
        if centerline is not None:
            csv_file = os.path.join(output_dir, filename.replace('.stl', '_centerline.csv'))
            # 儲存為CSV格式，包含標題行
            with open(csv_file, 'w') as f:
                f.write('x,y,z\n')  # 寫入標題行
                for point in centerline:
                    f.write(f'{point[0]},{point[1]},{point[2]}\n')
            print(f"  儲存中心線數據: {csv_file}")
            print(f"  中心線包含 {len(centerline)} 個點")
        
        print(f"完成處理: {filename}\n")

if __name__ == "__main__":
    input_dir = "/Users/julie/folder/vessel/organized/02_data_input/STL_smooth_solid"
    output_dir = "/Users/julie/folder/vessel/organized/03_centerline_auto"
    
    print("開始自動提取血管中心線...")
    process_vessels(input_dir, output_dir)
    print("\n所有檔案處理完成！")
    print(f"請開啟 {output_dir} 中的HTML檔案查看3D互動視覺化")