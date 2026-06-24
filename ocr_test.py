import easyocr

reader = easyocr.Reader(['ko', 'en'], gpu=True)

result = reader.readtext('./img/image02.png', detail=True, paragraph=False, decoder='greedy')

# detail 1: [([좌표], 텍스트, 확률)] 형태의 리스트를 반환., 0: 오직 텍스트만 리스트로 반환. (결과가 깔끔해서 단순 추출 시 유용)
# paragraph=True 인접한 텍스트 박스들을 하나의 문단으로 묶어서 반환합니다. 줄바꿈이 있는 글을 읽을 때 좋습니다.
# decoder='beamsearch' 정확도 높으나 느림
print(result)
for res in result:

    # res[0]: 글자 영역 좌표, res[1]: 인식된 텍스트, res[2]: 확신도(Confidence)
    print(f"Text: {res[1]} (Conf: {res[2]:.4f})")

'''
2. 이미지 처리 및 성능 최적화 (Image Processing)
이미지 인식의 정확도와 속도에 직접적인 영향을 줍니다.

mag_ratio (기본값: 1.0)

이미지 확대 비율입니다. 글자가 너무 작아서 인식이 안 될 때 1.5나 2.0으로 높이면 인식률이 올라가지만, 메모리 사용량도 함께 늘어납니다.

canvas_size (기본값: 2560)

인식 전 이미지를 리사이징할 최대 크기입니다. 아주 큰 이미지를 다룰 때 이 값을 조절하여 OOM(메모리 부족) 에러를 방지할 수 있습니다.

batch_size (기본값: 1)

GPU 사용 시 한 번에 처리할 이미지 조각의 개수입니다. GPU 메모리가 넉넉하다면 값을 키워 속도를 높일 수 있습니다.

3. 필터링 및 제한 (Constraints)
원하는 글자만 골라내거나, 특정 영역만 보고 싶을 때 유용합니다.

allowlist / blocklist

allowlist='0123456789': 숫자만 인식하도록 강제합니다. (계좌번호, 전화번호 추출 시 필수)

blocklist='...': 특정 특수문자나 단어를 제외하고 싶을 때 사용합니다.

min_size (기본값: 2)

너무 작은 텍스트 박스(노이즈 등)는 무시합니다. 픽셀 단위입니다.

contrast_ths / adjust_contrast

대비(Contrast) 임계값을 조절하여 너무 어둡거나 밝은 이미지에서의 인식률을 개선합니다.

'''