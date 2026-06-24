from pyhwpx import Hwp

hwp = Hwp()  # 한글 프로그램 실행 (백그라운드)
hwp.insert_text("안녕하세요, 파이썬에서 작성한 내용입니다.")
hwp.save_as("test.hwpx")
hwp.quit()