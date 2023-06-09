from cv2 import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

# crear base de datos
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f"{ruta}/{nombre}")
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])


# codificar imágenes
def codificar(imagenes):

    # crear una lista nueva
    lista_codificada = []

    # pasar todas las imagenes a RGB
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # codificar
        codificado = fr.face_encodings(imagen)[0]

        # agregar a la lista
        lista_codificada.append(codificado)

    # devolver lista codificada
    return lista_codificada


# registrar los ingresos
def registrar_ingresos(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registro = []

    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])

    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M')
        f.writelines(f"\n{persona}, {string_ahora}")


lista_empleados_codificada = codificar(mis_imagenes)

# tomar una imagen de camara web
captura = cv2.VideoCapture(0)

# leer imagen de la camara
# este metodo va a arrojar 2 elementos:
    # bool --> primero la informacion de si se ha podido realizar o no la captura
    # imagen --> y luego la imagen en si misma
exito, imagen = captura.read()

if not exito:
    print("No se ha podido relizar la captura")
else:
    # reconocer cara en captura
    cara_captura = fr.face_locations(imagen)

    # codificar cara capturada
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    # buscar coincidencias con la lista de imagenes de empleados
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        # como le entregamos una lista, nos devolvera tambien una lista (en los 2 casos)
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)
        # de la lista que de distancias que obtenemos, tenemos de identificar el valor inferior a 0.6
        indice_coincidencia = numpy.argmin(distancias)

        # mostrar coincidencias si las hay
        if distancias[indice_coincidencia] > 0.6:

            # mostrar el rectangulo
            y1, x2, y2, x1 = caraubic
            cv2.rectangle(img=imagen, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            # mostrar la imagen obtenida
            cv2.imshow('Imagen web', imagen)

            # imprimir mensaje
            print(distancias)
            print("Lo sentimos."
                  "No coincide con ninguno de nuestros empleados.")

            # mantener ventana abierta
            cv2.waitKey(4000)

        else:

            # buscar el nombre del empleado encontrado
            nombre = nombres_empleados[indice_coincidencia]

            # mostrar el rectangulo
            y1, x2, y2, x1 = caraubic
            cv2.rectangle(img=imagen, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
            cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # mostrar la imagen obtenida
            cv2.imshow('Imagen web', imagen)

            # llamar funcion registrar nombres
            registrar_ingresos(nombre)

            # imprimir mensaje
            print(distancias)
            print(f"!Bienvenido al trabajo {nombre}¡")

            # mantener ventana abierta
            cv2.waitKey(4000)



