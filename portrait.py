import cv2
from make_stringart import make_stringart
from make_stringart_with_preview import make_stringart_with_preview


res = (400, 400)
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # print(frame.shape)
    frame = frame[120:360, 240:500, :]
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (480, 480))
    # frame = frame[]

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("a"):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)

        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        stred = (frame.shape[0] // 2, frame.shape[1] // 2)
        velikost = min(frame.shape[0], frame.shape[1]) // 2
        frame = frame[
            stred[0] - velikost : stred[0] + velikost,
            stred[1] - velikost : stred[1] + velikost,
        ]

        frame = cv2.resize(frame, res)
        frame = cv2.resize(frame, (250, 250))

        out_img = make_stringart(
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
            32,
            1,
            resolution=(frame.shape[0], frame.shape[1]),
        )
        out_img = cv2.resize(out_img, (velikost * 2, velikost * 2))

        cv2.imshow("stringart", out_img)
        cv2.imshow("predloha", frame)


cap.release()
cv2.destroyAllWindows()
