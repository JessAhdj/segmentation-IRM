##################################################
#   Volumétrie du corps calleux sur images IRM   #
##################################################
# Auteur : Jessim Ahdjoudj
# Ce script permet à partir d'une seed choisie/placée par l'utilisateur :
#   - d’isoler une image du corps calleux 
#   - puis de faire un comptage de voxels pour la mesure de volume (càd sa volumétrie)

###################
#  Configuration  #
###################

# imports
import numpy as np 
import nibabel as nb 
import matplotlib.pyplot as plt 

# fonctions
def affiche(Im):
    plt.figure()
    plt.imshow(Im, cmap='gray', vmin=0,vmax=65536)
    plt.axis('off')
    plt.show() 

def affiche_binaire(Im):    
    plt.figure()
    plt.imshow(Im, cmap='gray')
    plt.axis('off')
    plt.show() 
    
def affiche_histogramme(Im):
    plt.figure()
    plt.hist(Im.flatten(), 256)
    plt.xlim([0, 65536])
    plt.ylim([0, 650])
    plt.show() 

def expansion_histogramme(Im):
    Ime = np.float64(Im)
    Ime = np.round(65536*(Ime-Ime.min())/(Ime.max()-Ime.min()))
    Ime = np.uint16(np.clip(Ime, 0, 65536))
    return Ime 

def inspection(I, info):
    print(info)
    print(I.min())
    print(I.max()) 
    print(I.dtype) 
    print('----')

def croissance_region(I,s,seuil):
    T = np.zeros([nx, ny], dtype=np.int8)
    p = [s[0],s[1]] 
    continuer = True
    while continuer : 
        T[p[0],p[1]] = 1 
        c = [p[0]-1,p[1]]
        if T[c[0],c[1]] == 0: 
            if abs(np.float64(I[c[0],c[1]]) - np.float64(I[s[0],s[1]])) <= np.float64(seuil):
                T[c[0],c[1]] = 3 
            else:
                T[c[0],c[1]] = 2 
        c = [p[0]+1,p[1]]
        if T[c[0],c[1]] == 0: 
            if abs(np.float64(I[c[0],c[1]]) - np.float64(I[s[0],s[1]])) <= np.float64(seuil):
                T[c[0],c[1]] = 3 
            else: 
                T[c[0],c[1]] = 2 
        c = [p[0],p[1]-1] 
        if T[c[0],c[1]] == 0: 
            if abs(np.float64(I[c[0],c[1]]) - np.float64(I[s[0],s[1]])) <= np.float64(seuil):
                T[c[0],c[1]] = 3 
            else: 
                T[c[0],c[1]] = 2 
        c = [p[0],p[1]+1]
        if T[c[0],c[1]] == 0: 
            if abs(np.float64(I[c[0],c[1]]) - np.float64(I[s[0],s[1]])) <= np.float64(seuil):
                T[c[0],c[1]] = 3 
            else: 
                T[c[0],c[1]] = 2
        t = np.where(T == 3) 
        if t[0].size == 0:
            continuer = False
        else:
            p[0] = t[0][0]
            p[1] = t[1][0]
    t = np.where(T == 2)
    for i in range(t[0].size):
        T[t[0][i],t[1][i]] = 0
    affiche_binaire(T) 
    I_hdr = Im1.header
    print(I_hdr.get_xyzt_units())
    print(I_hdr ['pixdim'] [1:4])
    volvox = np.prod(I_hdr ['pixdim'][1:4]) 
    print ('1 pixel =', volvox, 'mm3')
    nbvox = np.sum(T)
    print('Il y a', nbvox, 'voxels dans le corps calleux pour un volume de', nbvox*volvox, 'mm3')

#############
#  Analyse  #
#############

### SUJET 1 ###
print('\n--- Sujet 1 ---\n')
#1. ouvrir l'image 
S1 = 'sujet1.nii'
Im1 = nb.load(S1) 
Brain1 = Im1.get_fdata() 
nx, ny, nz = Brain1.shape

#2. inspection de l'image
inspection(Brain1, "Données image sujet 1")

#choisir les 2 coupes d'intérêt : 79, 80
Brain1 = Brain1[:,:,79:81]
# réorientation
Brain1 = np.rot90(Brain1,k=3)
Brain1 = np.fliplr(Brain1)

Sujet180 = Brain1[:,:,0]
Sujet181 = Brain1[:,:,1]

#3. pré-traitement 
Sujet180 = expansion_histogramme(Sujet180)
affiche_histogramme(Sujet180) # optionel, pour visualiser que l'histograme a bien été modifié

Sujet181 = expansion_histogramme(Sujet181)
affiche_histogramme(Sujet181) # optionel, pour visualiser que l'histograme a bien été modifié

#4. choix d'une seed
#je définis ma seed d'indices i,j
#je définis une autre image I~V où V en s = blanc pure
#je visualise V 
    #si ma seed appartient à ma région d'intérêt, je lance la croissance de région
    #sinon, je recommence pour une autre seed

seed = [120, 145]

V180 = np.copy(Sujet180)
V180[seed[0], seed[1]] = 0 # en noir ou 65535 en blanc
affiche(V180)
V181 = np.copy(Sujet181)
V181[seed[0], seed[1]] = 0 # en noir ou 65535 en blanc
affiche(V181)

#5. définition à la main du seuil
seuil = 7000 # sujet1

#6. on lance CR (I,s,seuil) = récupération d'une image correspond à la région d'intérêt
croissance_region(Sujet180,seed,seuil)
croissance_region(Sujet181,seed,seuil)

#7. visualisation ma région d'intérêt 
    #si besoin, on ajuste le seuil et on recommence

### SUJET 2 ###
print('\n--- Sujet 2 ---\n')
#1. ouvrir l'image 
S2 = 'sujet2.nii'
Im1 = nb.load(S2) 
Brain2 = Im1.get_fdata() 
nx, ny, nz = Brain2.shape

#2. inspection de l'image
inspection(Brain2, "Données image sujet 2")

#choisir les 2 coupes d'intérêt : 79, 80
Brain2 = Brain2[:,:,79:81]
# réorientation
Brain2 = np.rot90(Brain2,k=3)
Brain2 = np.fliplr(Brain2)

Sujet280 = Brain2[:,:,0]
Sujet281 = Brain2[:,:,1]

#3. pré-traitement 
Sujet280 = expansion_histogramme(Sujet280)
affiche_histogramme(Sujet280) # optionel, pour visualiser que l'histograme a bien été modifié

Sujet281 = expansion_histogramme(Sujet281)
affiche_histogramme(Sujet281) # optionel, pour visualiser que l'histograme a bien été modifié

#4. choix d'une seed
#je définis ma seed d'indices i,j
#je définis une autre image I~V où V en s = blanc pure
#je visualise V 
    #si ma seed appartient à ma région d'intérêt, je lance la croissance de région
    #sinon, je recommence pour une autre seed

seed2 = [105, 149]

V280 = np.copy(Sujet280)
V280[seed2[0], seed2[1]] = 0 # en noir ou 65535 en blanc
affiche(V280)
V281 = np.copy(Sujet281)
V281[seed2[0], seed2[1]] = 0 # en noir ou 65535 en blanc
affiche(V281)

#5. définition à la main du seuil
seuil2 = 8000 # sujet2

#6. on lance CR (I,s,seuil) = récupération d'une image correspond à la région d'intérêt
croissance_region(Sujet280,seed2,seuil2)
croissance_region(Sujet281,seed2,seuil2)

#7. visualisation ma région d'intérêt 
    #si besoin, on ajuste le seuil et on recommence

### SUJET 3 ###
print('\n--- Sujet 3 ---\n')
#1. ouvrir l'image 
S3 = 'sujet3.nii'
Im1 = nb.load(S3) 
Brain3 = Im1.get_fdata() 
nx, ny, nz = Brain3.shape

#2. inspection de l'image
inspection(Brain3, "Données image sujet 3")

#choisir les 2 coupes d'intérêt : 79, 80
Brain3 = Brain3[:,:,79:81]
# réorientation
Brain3 = np.rot90(Brain3,k=3)
Brain3 = np.fliplr(Brain3)

Sujet380 = Brain3[:,:,0]
Sujet381 = Brain3[:,:,1]

#3. pré-traitement 
Sujet380 = expansion_histogramme(Sujet380)
affiche_histogramme(Sujet380) # optionel, pour visualiser que l'histograme a bien été modifié

Sujet381 = expansion_histogramme(Sujet381)
affiche_histogramme(Sujet381) # optionel, pour visualiser que l'histograme a bien été modifié

#4. choix d'une seed
#je définis ma seed d'indices i,j
#je définis une autre image I~V où V en s = blanc pure
#je visualise V 
    #si ma seed appartient à ma région d'intérêt, je lance la croissance de région
    #sinon, je recommence pour une autre seed

seed3 = [95, 148]

V380 = np.copy(Sujet380)
V380[seed3[0], seed3[1]] = 0 # en noir ou 65535 en blanc
affiche(V380)
V381 = np.copy(Sujet381)
V381[seed3[0], seed3[1]] = 0 # en noir ou 65535 en blanc
affiche(V381)

#5. définition à la main du seuil
seuil3 = 6700 # sujet3

#6. on lance CR (I,s,seuil) = récupération d'une image correspond à la région d'intérêt
croissance_region(Sujet380,seed3,seuil3)
croissance_region(Sujet381,seed3,seuil3)

#7. visualisation ma région d'intérêt 
    #si besoin, on ajuste le seuil et on recommence
