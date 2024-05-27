from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import sys
from utils import create_compound_image
def findRealCoordinates(corners): #coordinate img
    square_size = 11
    real_coordinates = np.empty_like(corners) #array vuoto di grandezza uguale a corners

    for index, corner in enumerate(corners):
        u_coord = corner[0]
        v_coord = corner[1]

        grid_size_cv2 = tuple(reversed(grid_size))
        u_index, v_index = np.unravel_index(index, grid_size_cv2) #trasforma il valore index nelle coordinate della matrice grid_size
        x_mm = (u_index) * square_size #distanza dell'angolo rispetto al valore di reference 0,0
        y_mm = (v_index) * square_size

        real_coordinates[index,:] = [x_mm, y_mm]
    return real_coordinates

def estimateHomography(corners, real_coordinates):
    A = np.empty((0,9), dtype=float) #A è la matrice tc Ax = 0 con x l'incognita

    for index, corner in enumerate(corners):
        Xpixel = corners[index, 0]
        Ypixel = corners[index, 1]
        Xmm = real_coordinates[index, 0]
        Ymm = real_coordinates[index, 1]

        m = np.array([Xmm, Ymm, 1]).reshape(1, 3)
        O = np.array([0, 0, 0]).reshape(1, 3)

        # Construct A
        A = np.vstack((A, np.hstack((m, O, -Xpixel * m))))
        A = np.vstack((A, np.hstack((O, m, -Ypixel * m))))

    U, S, Vtransposed = np.linalg.svd(A) #questo serve per scomporre la matrice in U * S * V 
    h =Vtransposed.transpose()[:,-1] #ci serve V trasposta e solo l'ultima colonna
    H = h.reshape(3,3) #da un array 9 a un 3x3
    #H è la nostra omografia!
    return H

def findVijValue(H,i,j): #calcoliamo il valore vij di V
    Vij = np.empty(6)
    Vij[0] = H[0][i]*H[0][j]
    Vij[1] = H[0][i]*H[1][j] + H[1][i]*H[0][j]
    Vij[2] = H[1][i]*H[1][j]
    Vij[3] = H[2][i]*H[0][j] + H[0][i]*H[2][j]
    Vij[4] = H[2][i]*H[1][j] + H[1][i]*H[2][j]
    Vij[5] = H[2][i]*H[2][j]
    return Vij

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#carica immagini
folderpath= 'images/'
images_path = [os.path.join(folderpath, imagename) for imagename in os.listdir(folderpath) if imagename.endswith(".tiff")]
images_path.sort()
grid_size = (8,11)
homographies = []
for p in images_path:
    corners = []
    im = cv2.imread(p)
    return_value, corners = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
    if not return_value:
        print(f"pattern not found for image {p}")
    corners=corners.reshape((88,2)).copy()
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001) # tuple for specifying the termination criteria of the iterative refinement procedure cornerSubPix()
    #corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
    real_coordinates = findRealCoordinates(corners)
    H = estimateHomography(corners, real_coordinates)
    homographies.append(H)
#adesso che abbiamo l'omografia H, dobbiamo trovare P
#per farlo seguiamo la procedura di zhang con Vb=0

#Calcoliamo V
V = []
for h in homographies:
    #2 equazioni per ogni omografia 
    v = []
    v.append(findVijValue(h,0,1))
    v11= findVijValue(h,0,0)
    v22= findVijValue(h,1,1)
    v.append(np.subtract(v11,v22))
    V.append(v)

#Scomposizione UESt della matrice V
V = np.array(V)
V = V.reshape(40,6)
U,E,Stransposed = np.linalg.svd(V)
S = Stransposed.transpose()
#la sol di Vb=0 è l'ultima colonna di S
b = S[:,-1]
B = np.empty((3,3))
B[0]= [b[0],b[1],b[3]]
B[1]= [b[1],b[2],b[4]]
B[2]= [b[3],b[4],b[5]]

#Adesso che abbiamo B, dobbiamo farne la scomposizione di cholesky
#se B non è pos def e dato che è definita per uno scale factor, possiamo moltiplicare B per -1

if not is_pos_def(B):
    B = -1 * B

L= np.linalg.cholesky(B) #lower triangular
Lt = L.transpose()

#La calibration matrix è l'inversa della trasposta

K = np.linalg.inv(Lt)
#K(3,3) viene leggermente sopra 1, quindi lo settiamo a 1
print("K=",K)
K[2][2] = 1
#adesso calcoliamo r e t per ogni immagine
rtMatrix = []
count=0
for h in homographies:
    print(h)
    h1 = np.array(h[:,0])
    h2 = np.array(h[:,1])
    h3 = np.array(h[:,2])
    denonim = np.linalg.norm(np.matmul(np.linalg.inv(K),h1.transpose()))
    lambd = 1/denonim
    r1 = lambd *(np.matmul(np.linalg.inv(K),h1.transpose()))
    r2 = lambd *(np.matmul(np.linalg.inv(K),h2.transpose()))
    r3 = np.cross(r1,r2)
    t = lambd *(np.linalg.inv(K)@h3)
    rtMatrix.append([r1,r2,r3,t])
    count+=1
rtMatrix= np.array(rtMatrix)

#reprojection error, prendiamo l'immagine image03.tiff
totRepErr= 0
immagine = cv2.imread(folderpath + "image05.tiff")
paramIntrinsechi = np.array(rtMatrix[5]).transpose()
P = K@paramIntrinsechi
return_value, puntipixel = cv2.findChessboardCorners(immagine, patternSize=grid_size, corners=None)
puntipixel= puntipixel.reshape((88,2)).copy()
coordReali =  findRealCoordinates(puntipixel)
print(puntipixel)
#for pixel in puntipixel:
    #immagine = cv2.circle(immagine, (int(pixel[0]),int(pixel[1])), radius=3, color=(0, 255, 0), thickness=1)
for i, coord in enumerate(coordReali):
    #troviamo le coordinate omogenee
    homog = np.array([coord[0],coord[1],0,1])
    #proiettiamo
    project_hom = P@homog
    projections = np.empty(2)
    projections[0]= int(project_hom[0]/project_hom[2])
    projections[1] = int(project_hom[1]/project_hom[2])
    immagine = cv2.circle(immagine, (int(projections[0]),int(projections[1])), radius=7, color=(255, 0, 0), thickness=1)
    immagine = cv2.circle(immagine, (int(puntipixel[i][0]),int(puntipixel[i][1])), radius=7, color=(0, 255, 0), thickness=1)
    totRepErr += (projections[0]-puntipixel[i][0])**2 + (projections[1]-puntipixel[i][1])**2
print(totRepErr)
plt.imshow(immagine)
plt.show()

#superimposing di un rettangolo
rect_width = 99
rect_height = 44

bottom_left_corner_x = -11
bottom_left_corner_y = 33
superimposed = []
for i,im in enumerate(images_path):
    #base rettangolo
    image = cv2.imread(im)
    overlay = image.copy()
    vx = bottom_left_corner_x + np.array([0, 0, rect_width, rect_width])
    vy = bottom_left_corner_y + np.array([0, rect_height, rect_height, 0])
    homogeneousIN = np.vstack((vx, vy, np.ones_like(vx)))
    homogeneousOUT = homographies[i] @ homogeneousIN #moltiplicazione tra mappa e input
    xyOUT = (homogeneousOUT/homogeneousOUT[2])[:2] #in questo caso stiamo dividendo per la terza variabile e teniamo le prime 2
    cv2.polylines(image,np.int32([xyOUT.transpose()]), isClosed=True, color=(0,0,0),thickness=4)
    cv2.fillPoly(overlay,np.int32([xyOUT.transpose()]),color=(255,0,0))
    #parte rialzata
    height = 25
    homogeneousIN2 = np.vstack((vx, vy, np.full_like(vx,height),np.ones_like(vx)))
    paramIntrinsechi = np.array(rtMatrix[i])
    P = K@paramIntrinsechi.transpose()
    homogeneousOUT2 = P @homogeneousIN2
    xyzOUT = (homogeneousOUT2/homogeneousOUT2[2])[:2]
    cv2.polylines(image,np.int32([xyzOUT.transpose()]), isClosed=True, color=(0,0,0),thickness=4)
    cv2.fillPoly(overlay,np.int32([xyzOUT.transpose()]), color=(0,255,0))
    image_new = cv2.addWeighted(overlay, 0.4, image, 1 - 0.4, 0)
    superimposed.append(image_new)




compound_image=create_compound_image(rows=4,cols=5,limages=superimposed)
plt.figure(figsize=(20, 20))
plt.imshow(compound_image)
plt.show()




    

