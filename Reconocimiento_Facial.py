from cv2 import cv2
import face_recognition as fr

# cargar imagenes
foto_control = fr.load_image_file('Fotos_Prueba/FotoA.jpg')
foto_prueba = fr.load_image_file('Fotos_Prueba/FotoB.jpg')

# pasar imagenes a RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# localizar cara control (dentro de la imagen)
lugar_cara_A = fr.face_locations(foto_control)[0]
# hacer que el sistema entienda a traves de numeros donde se encuentra lo que necesitamos
cara_codificada_A = fr.face_encodings(foto_control)[0]

# localizar cara prueba (dentro de la imagen)
lugar_cara_B = fr.face_locations(foto_prueba)[0]
# hacer que el sistema entienda a traves de numeros donde se encuentra lo que necesitamos
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

# mostrar rectangulos
'''
print(lugar_cara_A)
print(lugar_cara_B)
'''
cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0, 255, 0),
              thickness=2)

cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0, 255, 0),
              thickness=2)

# relizar comparacion
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)
# modificando la tolerancia
# resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B, tolerance=0.3)
# resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B, tolerance=0.9)


print(resultado)

# mostrar imagenes
# no seria necesario pero es para aprender a como usar cv2 para mostrar imagenes
cv2.imshow('Foto control', foto_control)
cv2.imshow('Foto prueba', foto_prueba)

# medida de la distancia
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)
print(distancia)

# mantener programa abierto
cv2.waitKey(0)




