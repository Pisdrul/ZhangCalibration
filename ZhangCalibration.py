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
    np.linalg.norm( A - U[:,:9]@np.diag(S)@Vtransposed )
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
filepath = folderpath + "image02.tiff"
image = cv2.imread(filepath)
grid_size = (8,11)
#trova corners (faccio prima senza refinement, dopo con)
return_value, corners = cv2.findChessboardCorners(image, patternSize=grid_size)
corners=corners.reshape((88,2)).copy()


real_coordinates = findRealCoordinates(corners)
H = estimateHomography(corners, real_coordinates)


#adesso che abbiamo l'omografia H, dobbiamo trovare P
#per farlo seguiamo la procedura di zhang con Vb=0

#Calcoliamo V
V = np.empty((2,6))
V[0]=findVijValue(H,0,1)
V11= findVijValue(H,0,0)
V22= findVijValue(H,1,1)
V[1] = np.subtract(V11,V22)
print(V)

#Scomposizione UESt della matrice V
U,E,Stransposed = np.linalg.svd(V)
S = Stransposed.transpose()
#la sol di Vb=0 è l'ultima colonna di S
b = S[:,-1]
B = np.empty((3,3))
B[0]= [b[0],b[1],b[3]]
B[1]= [b[1],b[2],b[4]]
B[2]= [b[3],b[4],b[5]]
#guarda meglio qua che dovrebbe avere i valori in diagonale dello stesso segno