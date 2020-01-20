import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def whitening(gray):
    flat = gray.flatten()
    mean = np.mean(flat)
    deviation = np.std(flat)
    print(mean,deviation)

    def f(x):
        return float((x - mean)/(deviation))

    trf = np.vectorize(f)
    new_gray = trf(flat)

    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         if(gray[i][j]!=new_gray[i][j]):
    #             print("NIM")

    new_gray = np.reshape(new_gray,gray.shape)
    
    # cv2.imshow("Whitening",new_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("beach_gray_whitening.png",new_gray)
    # fig = plt.figure(figsize=(1,2))
    # fig.add_subplot(1,2,1)
    # plt.imshow(new_gray,cmap="gray")
    # fig.add_subplot(1,2,2)
    # plt.imshow(gray,cmap="gray")
    # plt.show()
    # if(gray.all() == new_gray.all()):
    #     print("LOL")
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    fig.add_subplot(1,2,1)
    plt.imshow(gray, cmap='gray')

    # display the new image
    fig.add_subplot(1,2,2)
    plt.imshow(new_gray, cmap='gray')

    plt.show(block=True)

def histogram_equalization(gray):
    flat = gray.flatten()
    plt.hist(flat,bins=50)
    plt.show()

    hist = np.zeros(256)
    for pixel in flat:
        hist[pixel] +=1
    plt.plot(hist)

    def cumsum(a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    cs = cumsum(hist)
    plt.plot(cs)   

    nj = (cs - cs.min())*255
    N = cs.max() - cs.min()

    cs = nj/N
    plt.plot(cs)
    cs = cs.astype('uint8')

    img_new = cs[flat]
    plt.hist(img_new,bins=50)

    img_new = np.reshape(img_new, gray.shape)
    # cv2.imshow("New image",img_new)
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    fig.add_subplot(1,2,1)
    plt.imshow(gray, cmap='gray')

    # display the new image
    fig.add_subplot(1,2,2)
    plt.imshow(img_new, cmap='gray')

    plt.show(block=True)



img = cv2.imread("./chess.jpeg")
print(type(img),img.shape)

cv2.imshow("Preview", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("cube_gray.png",gray)
# print(gray[100][100])
# gray = gray.astype(float)

choice = 0
print("Choose 1. Whitening 2. Histogram Equalization 3. Both")
choice = input()
if(choice == "1"):
    whitening(gray)
elif (choice == "2"):
    histogram_equalization(gray)
elif (choice == "3"):
    result = whitening(gray)
    histogram_equalizaiton(result)
else:
    exit()


# def histogram_equalization(gray):






# Extra Code
# mean = 0
# temp = 0
# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         temp+=gray[i][j]
#     temp/=(gray.shape[0]*gray.shape[1])
#     mean = temp
#     temp=0

# variance = 0
# temp = 0
# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         temp+=(gray[i][j]*mean)*(gray[i][j]*mean)
#     temp/=(gray.shape[0]*gray.shape[1])
#     variance = temp
#     temp=0

# new_gray = copy.deepcopy(gray)
# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         new_gray[i][j] = (gray[i][j] - mean)/math.sqrt(variance)