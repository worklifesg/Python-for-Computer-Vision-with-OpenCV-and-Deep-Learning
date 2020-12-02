#cv2.imshow works in Jupyter notebook but not in Colab
import cv2

img = cv2.imread('00-puppy.jpg')

while True:
    cv2.imshow('Puppy',img)
    if cv2.waitKey(1) & 0xFF == 27: # 0xFF is a hexadecimal constant which is 1111111
        break

    # if we waited at least 1ms AND we have pressed Esc (27) / ord(q) - Q key

cv2.destroyAllWindows()
