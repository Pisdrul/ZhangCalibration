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
    print(H)
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
    #trova corners (faccio prima senza refinement, dopo con)
    print(p)
    im = cv2.imread(p)
    return_value, corners = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
    if not return_value:
        print(f"pattern not found for image {p}")
    corners=corners.reshape((88,2)).copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001) # tuple for specifying the termination criteria of the iterative refinement procedure cornerSubPix()
    corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
    real_coordinates = findRealCoordinates(corners)
    H = estimateHomography(corners, real_coordinates)
    homographies.append(H)

print(homographies)
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
    print(v)
    V.append(v)

#Scomposizione UESt della matrice V
V = np.array(V)
V = V.reshape(40,6)
U,E,Stransposed = np.linalg.svd(V)
S = Stransposed.transpose()
#la sol di Vb=0 è l'ultima colonna di S
b = S[:,-1]
print(S)
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
print(L)
print(Lt)

#La calibration matrix è l'inversa della trasposta

K = np.linalg.inv(Lt)
#K(3,3) viene leggermente sopra 1, quindi lo settiamo a 1
K[2][2] = 1
print(K)
#adesso calcoliamo r e t per ogni immagine
rtMatrix = []
count=0
for h in homographies:
    print(h)
    h1 = h[:,0]
    h2 = h[:,1]
    h3 = h[:,2]
    denonim = np.linalg.norm(np.linalg.inv(K)@h1)
    lambd = 1/denonim
    r1 = lambd *(np.linalg.inv(K)@h1)
    r2 = lambd *(np.linalg.inv(K)@h2)
    r3 = np.cross(r1,r2)
    t = r1 = lambd *(np.linalg.inv(K)@h3)
    rtMatrix.append([r1,r2,r3,t])
    print(rtMatrix[count])
    count+=1
rtMatrix= np.array(rtMatrix)
print(rtMatrix)

#reprojection error, prendiamo l'immagine image03.tiff
totRepErr= 0
immagine = cv2.imread(folderpath + "image03.tiff")
paramIntrinsechi = np.array(rtMatrix[3]).transpose()
print(rtMatrix[3])
print(rtMatrix[3].transpose())
P = K@paramIntrinsechi
return_value, puntipixel = cv2.findChessboardCorners(immagine, patternSize=grid_size, corners=None)
puntipixel= puntipixel.reshape((88,2)).copy()
coordReali =  findRealCoordinates(puntipixel)
print(coordReali)
for coord in coordReali:
    #troviamo le coordinate omogenee
    homog = np.array([coord[0],coord[1],0,1])
    #proiettiamo
    project_hom = P@homog
    projections = np.empty(2)
    projections[0]= int(project_hom[0]/project_hom[2])
    projections[1] = int(project_hom[1]/project_hom[2])
    immagine = cv2.circle(immagine, (int(projections[0]),int(projections[1])), radius=4, color=(255, 0, 0), thickness=1)
plt.imshow(immagine)
plt.show()



    

