import pandas as pd
import numpy as np

def calculate_geometry(H, x, z):
    """Tính diện tích mặt cắt ướt (A) và chu vi ướt (P) tại mực nước H"""
    A = 0.0
    P = 0.0
    
    for i in range(len(x) - 1):
        x1, z1 = x[i], z[i]
        x2, z2 = x[i+1], z[i+1]
        
        # Nếu cả 2 điểm đều chìm dưới nước
        if z1 < H and z2 < H:
            # Diện tích hình thang
            h1 = H - z1
            h2 = H - z2
            dx = x2 - x1
            A += 0.5 * (h1 + h2) * dx
            # Chiều dài cạnh đáy
            dz = z2 - z1
            P += np.sqrt(dx**2 + dz**2)
            
        # Nếu điểm 1 chìm, điểm 2 nổi (cắt bờ)
        elif z1 < H and z2 >= H:
            dx = x2 - x1
            dz = z2 - z1
            # Nội suy tìm tọa độ X tại vị trí cắt mặt nước
            x_intersect = x1 + dx * (H - z1) / dz
            h1 = H - z1
            dx_wetted = x_intersect - x1
            A += 0.5 * h1 * dx_wetted
            P += np.sqrt(dx_wetted**2 + (H - z1)**2)
            
        # Nếu điểm 1 nổi, điểm 2 chìm (cắt bờ)
        elif z1 >= H and z2 < H:
            dx = x2 - x1
            dz = z2 - z1
            # Nội suy tìm tọa độ X tại vị trí cắt mặt nước
            x_intersect = x1 + dx * (H - z1) / dz
            h2 = H - z2
            dx_wetted = x2 - x_intersect
            A += 0.5 * h2 * dx_wetted
            P += np.sqrt(dx_wetted**2 + (H - z2)**2)
            
    return A, P

def main():
    # 1. Đọc dữ liệu (Thay đổi tên cột nếu file của em khác)
    print("Đang đọc dữ liệu...")
    df_dem = pd.read_excel("D:\HEC-RAS-Projects\Elevation.xlsx")
    x = df_dem['X'].values
    z = df_dem['Z'].values
    
    df_water = pd.read_excel(r"D:\Downloands\Tableau-Crues1926 (1).xlsx")
    # Giả sử cột ngày là 'Date' và cột mực nước là 'H'
    
    # 2. Thiết lập thông số Manning (Cần điều chỉnh theo thực tế)
    n = 0.035      # Hệ số nhám sông tự nhiên
    S = 0.0001     # Độ dốc mặt nước/đáy sông
    
    # 3. Tính toán lặp cho từng ngày
    print("Đang tính toán lưu lượng...")
    discharge_list = []
    
    for index, row in df_water.iterrows():
        H_day = row['H']  # Đọc mực nước ngày đó
        A, P = calculate_geometry(H_day, x, z)
        
        if P > 0:
            R = A / P
            # Áp dụng công thức Manning
            Q = (1 / n) * A * (R ** (2/3)) * (S ** 0.5)
        else:
            Q = 0
            
        discharge_list.append(Q)
        
    # 4. Lưu kết quả
    df_water['Q_calculated'] = discharge_list
    df_water.to_csv('Discharge_1926_Result.csv', index=False)
    print("Hoàn tất! Đã lưu kết quả vào file 'Discharge_1926_Result.csv'")

if __name__ == "__main__":
    main()