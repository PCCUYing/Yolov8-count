from ultralytics import YOLO
import cv2, time, numpy as np
from shapely.geometry import Polygon

# 設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target = 'testfile/Counter.mp4'
model = YOLO('yolov8m.pt')  # 預設模型：n, s, m, l, x 五種大小

names = model.names  # 認識的80物件 字典：編號及名稱
print(names)

# 區域三維陣列
area = [
    [[1040, 173], [1054, 359], [1067, 466], [1081, 597], [1112, 801], [1123, 928], [1357, 921], [1299, 670],
     [1233, 425], [1216, 242], [1195, 177], [1185, 146], [1071, 156]]  # 車道1
]

# 繪製區域   影像, 區域座標, 顏色, 寬度
def drawArea(f, area, color, th):
    for a in area:
        v = np.array(a, np.int32)
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)
    return f

# 取得重疊比例 引用 物件、區域
def inarea(object, area):
    inAreaPercent = []  # area陣列，物件在所有區域的比例
    # 把物件座標變成多邊形  [x1, y1]左上,[x2, y1]右上      [x2, y2]右下                  [x1, y2]左下   
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]]
    for i in range(len(area)):
        poly1 = Polygon(b)
        poly2 = Polygon(area[i])
        intersection_area = poly1.intersection(poly2).area  # 重疊區域部分面積 = 交集部分面積
        poly1Area = poly1.area  # 物件區域 我自己的面積        
        overlap_percent = (intersection_area / poly1Area) * 100  # 重疊比例
        inAreaPercent.append(overlap_percent)  # 加入比例
    return inAreaPercent

# 追蹤物件的資料結構
tracked_objects = {}  # 格式: {object_id: {"start_time": float, "last_seen": float}}

cap = cv2.VideoCapture(target)

while 1:
    st = time.time()
    r, frame = cap.read()
    if not r:  # 讀取失敗
        break
    results = model(frame, verbose=False)  # YOLO辨識verbose=False不顯示文字結果

    frame = drawArea(frame, [area[0]], (0, 255, 0), 3)

    carCount = [0, 0, 0]  # 初始汽車數量
    current_time = time.time()

    for box in results[0].boxes.data:
        x1 = int(box[0])  # 左
        y1 = int(box[1])  # 上
        x2 = int(box[2])  # 右
        y2 = int(box[3])  # 下
        r = round(float(box[4]), 2)  # 信任度
        n = names[int(box[5])]  # 名字

        # 如果不是汽車、巴士、卡車 的話就跳過
        if n not in ['person'] or r < 0.5:
            continue  # 下一個

        # 計算物件是否進入區域
        tempObj = [x1, y1, x2, y2, r, n]
        ObjInArea = inarea(tempObj, area)

        # 物件在區域0的比例>=25
        if ObjInArea[0] >= 25:
            # 計算物件中心點
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            object_id = f"{center_x}-{center_y}"  # 使用中心點作為唯一ID

            if object_id not in tracked_objects:
                # 第一次進入範圍，記錄進入時間
                tracked_objects[object_id] = {"start_time": current_time, "last_seen": current_time}
            else:
                # 更新最後看到的時間
                tracked_objects[object_id]["last_seen"] = current_time

            # 計算在範圍內的時間 (秒)
            time_in_area = current_time - tracked_objects[object_id]["start_time"]

            # 畫出框並標註時間
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{n} {r:.2f} {time_in_area:.1f}s", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
            carCount[1] += 1

    # 白色背景
    cv2.rectangle(frame, (0, 0), (200, 200), (255, 255, 255), -1)

    # 區域1
    cv2.putText(frame, 'Area1=' + str(carCount[1]), (20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # 移除超時未見的物件
    timeout = 5  # 設定超時時間，例如5秒
    for obj_id in list(tracked_objects.keys()):  # 使用 list 遍歷避免動態修改字典
        if current_time - tracked_objects[obj_id]["last_seen"] > timeout:
            del tracked_objects[obj_id]  # 超時未見，移除物件

    et = time.time()
    FPS = round((1 / (et - st)), 1)
    cv2.putText(frame, 'FPS=' + str(FPS), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
