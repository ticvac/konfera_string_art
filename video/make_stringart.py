import cv2
import numpy as np
import math
from bresenham import bresenham as line_iter
from matplotlib import pyplot as plt


def make_stringart(
    img: np.ndarray, color: int, thickness: int, resolution: tuple = (500, 500)
):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resolution)

    # orig = img.copy()

    canvas = np.full_like(img, 255)
    # canvas[0, 0] = 0
    # canvas[0, 1] = 255

    def create_circular_mask(h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if (
            radius is None
        ):  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    # preprocess
    mask = create_circular_mask(img.shape[0], img.shape[1])
    imgcircle = img.copy()
    imgcircle[~mask] = 255
    imginverted = np.full(img.shape, 255) - imgcircle

    orig = img.copy()
    mask = create_circular_mask(orig.shape[0], orig.shape[1])
    orig[~mask] = 255
    stopping_point = np.mean(orig) * 1.1

    img = imginverted

    def generate_circle_points(center, radius, n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = [
            (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
            for angle in angles
        ]
        return points

    center = (img.shape[0] // 2, img.shape[1] // 2)
    radius = min(img.shape) // 2 - 5
    n = 200
    circle_points = generate_circle_points(center, radius, n)

    # Plot the points on the image
    # img_gray = np.full_like(img, 0, dtype=np.uint8)
    # for point in circle_points:
    #     img_gray[int(point[1]), int(point[0])] = 255
    #     cv2.line(
    #         img_gray,
    #         (int(point[1]), int(point[0])),
    #         (int(point[1]), int(point[0])),
    #         255,
    #         1,
    #     )

    # imshow(img_gray, cmap="gray")

    spagat = [0]

    while True:
        from_pos = circle_points[spagat[-1]]
        best = [-1, -math.inf]
        for indx, to_pos in enumerate(circle_points):
            if indx == spagat[-1]:
                continue

            lajna = []

            for x, y in line_iter(
                int(from_pos[0]), int(from_pos[1]), int(to_pos[0]), int(to_pos[1])
            ):
                # print("herke")
                lajna.append(img[y, x])
                # brehem

            # print(lajna)
            lajna = np.array(lajna)

            error = np.sum(lajna)
            # print(error)

            if error > best[1]:
                # print(error, indx)
                best = [indx, error]
                bst = (from_pos, to_pos)
                best_lajna = lajna

            # break

        canvas_w_line = cv2.line(
            np.full_like(canvas, 0),
            tuple(map(int, from_pos)),
            tuple(map(int, circle_points[best[0]])),
            color,
            thickness,
        )

        canvas = cv2.subtract(canvas, canvas_w_line)
        img = cv2.subtract(img, canvas_w_line, dtype=cv2.CV_8U)

        spagat.append(best[0])
        start = best[0]

        # print(np.mean(canvas))
        if np.mean(canvas) < stopping_point:
            break

    # print(np.mean(canvas))
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # imshow(
    #     canvas,
    #     cmap="gray",
    # )
    # plt.subplot(1, 2, 2)
    # imshow(orig, cmap="gray")

    return canvas


if __name__ == "__main__":
    img = cv2.imread("honza.png")
    string_img = make_stringart(img, 40, 2, resolution=(250, 250))
    plt.imshow(string_img, cmap="gray")
    plt.show()
