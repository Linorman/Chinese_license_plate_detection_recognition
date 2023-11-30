import cv2

from detect_plate import process_image

if __name__ == "__main__":
    img = "imgs\\double_yellow.jpg"
    model_path = "weights\\plate_detect.pt"
    plate_rec_model = "weights\\plate_rec_color.pth"
    result = process_image(img, model_path, plate_rec_model)
