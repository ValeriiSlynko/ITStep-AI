# застосування -- усунення шуму
import cv2
import utils


# img = cv2.imread("data/lesson3/castello_blurred.png")
# img = cv2.resize(img, (600, 600))
#
# cv2.imshow("orig", img)
#
# noise = utils.add_gaussian_noise(img, 0, 20)
# cv2.imshow("noised", noise)
#
#
#
# # Гаусове розмиття
# gauss = cv2.GaussianBlur(
#     noise,  # зображення з шумом
#     (7, 7),   # розмір фільтру(ядра)
#     sigmaX=5.5,    # наскільки важливими є далекі пікселі
# )
#
#
# cv2.imshow("gauss", gauss)
#
#
#
# двосторонній фільтр
# bilat = cv2.bilateralFilter(
#     noise,  # зображення з шумом
#     d=9,    # розмір фільтру
#     sigmaColor=75,   # наскільки важливі пікселі іншого кольору
#     sigmaSpace=75,   # наскільки важливими є далекі пікселі
# )
#
# cv2.imshow("bilat", bilat)
#
#
#
# nlmean = cv2.fastNlMeansDenoisingColored(noise, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
#
#
# cv2.imshow("nlmean", nlmean)



# бінарізація

img = cv2.imread("data/lesson3/darken_page.jpg")

cv2.imshow("orig", img)

# зображення має бути чорно біле
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)

# проста бінарізація

threshold = 10  # поріг для чорного

mask = gray < threshold

gray[mask] = 0  # все що менше 50 вважаємо чорним
gray[~mask] = 255  # все що більше 50 вважаємо білим

cv2.imshow("simple bin",  gray)



# адаптивна бінарізація

res = cv2.adaptiveThreshold(
    gray,   # зображення з текстом(чорнобіле)
    255,    #  білий колір
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # фільтр для обрахунку порогу(гаус)
    cv2.THRESH_BINARY,   # це просто треба вказати
    21,   # розмір фільтру
    2,          # наскільки піксель має відрізнятися від порогу
)


cv2.imshow("adaptive", res)





cv2.waitKey(0)