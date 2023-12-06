import json
import os

# 바운딩 형식에서 yolo형식으로 변환
# x = 바운딩 박스 가로축 시작점
# y = 바운딩 박스 세로축 시작점
# w, h = 바운딩 박스 너비, 높이
# yolo 형식 = x,y 값은 바운딩 박스의 중앙 => 바운딩 박스의 정중앙 좌표를 각각 이미지 너비와 높이로 나눔
# w, h값은 각각 이미지 너비, 높이로 나눔
def yolo_format(bbox):
    x, y, width, height = bbox
    center_x = ((x + (width/2))/976)
    center_y = ((y + (height/2))/1280)
    box_width = width/976
    box_height = height/1280
    return 1, center_x, center_y, box_width, box_height

def json_files(directory_path, output_directory):
    # 디렉토리의 모든 JSON 파일에 대해 반복
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as json_file:
            # JSON 파일 읽기
            json_data = json.load(json_file)

            # 바운딩 박스 좌표를 YOLO 형식으로 변환
            bbox_coordinates = json_data["annotations"][0]["bbox"]
            yolo_coordinates = yolo_format(bbox_coordinates)

            # YOLO 형식으로 변환된 좌표를 해당 JSON 파일명과 동일한 이름의 TXT 파일에 저장
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file_path, 'w') as output_file:
                output_file.write(' '.join(map(str, yolo_coordinates)))
                        
# JSON 파일이 있는 디렉토리와 출력 TXT 파일 디렉토리 설정
input_directory = r'C:\Python\K-039147_json'
output_directory = r'C:\Python\K-039147_txt'                       

# 모든 JSON 파일 처리 및 YOLO 형식으로 변환된 좌표를 파일에 저장
json_files(input_directory, output_directory)             

        
# txt 파일 내용 확인
f = open("C:\Python\K-039123_txt", 'r')
lines = f.readlines()
for line in lines:
    print(line)
f.close()