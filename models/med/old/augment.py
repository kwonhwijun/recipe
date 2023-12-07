import os
import imgaug.augmenters as iaa
import imageio

# 이미지 경로
input_folder = r"C:\Python\augtest"
# image = imageio.imread(r"C:\Python\augtest")

# 결과 이미지를 저장할 상위 폴더 경로
output_folder = r"C:\test1\recipe\augresult"

# 증강
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 좌우 반전
    iaa.GaussianBlur(sigma=(0, 3.0)),  # 가우시안 블러
    iaa.Crop(percent=(0, 0.1)),  # 임의의 크기로 자르기
    iaa.Affine(rotate=(-45, 45))  # 임의의 각도로 회전
])

# 블러 제거
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 좌우 반전
    iaa.Crop(percent=(0, 0.1)),  # 임의의 크기로 자르기
    iaa.Affine(rotate=(-45, 45))  # 임의의 각도로 회전
])

# 이미지 폴더 내의 모든 이미지 파일에 대해 처리
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)

        image = imageio.imread(image_path)
        
        # 결과 이미지를 저장할 폴더 경로 생성
        result_folder = os.path.join(output_folder, f"result_{filename.split('.')[0]}")
        os.makedirs(result_folder, exist_ok=True)
        
        # 이미지를 10번 증강 후 저장
        for i in range(1, 11):
            augmented_image = seq.augment_image(image)
            
            # 결과 이미지 저장
            output_path = os.path.join(result_folder, f'{filename.split(".")[0]}_{i}.png')
            imageio.imwrite(output_path, augmented_image)
            
            print(f'Augmented image saved to: {output_path}')