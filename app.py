import streamlit as st
import functions as fn
import matplotlib.pyplot as plt
from skimage import io

# Inicializa la lista de imágenes en el estado de la sesión si aún no existe
if 'Imagenes' not in st.session_state:
    st.session_state.Imagenes = []

if 'ImagenesFiltradas' not in st.session_state:
    st.session_state.ImagenesFiltradas = []

st.title("Mi Aplicación Streamlit")

if st.button('Descargar'):
    # Actualiza la lista de imágenes en el estado de la sesión
    st.session_state.Imagenes = fn.descarga(st.session_state.Imagenes)
    for img in st.session_state.Imagenes:
        plt.figure()  # Crea una nueva figura para cada imagen
        plt.imshow(img)
        plt.axis('off')
        st.pyplot()

filtro = st.number_input('Ingrese el filtro deseado (0 a 14)', min_value=0, max_value=14, value=0)

# Utiliza la lista de imágenes desde el estado de la sesión
if len(st.session_state.Imagenes) > 0:
    if st.button('Multiprocessing'):
        st.session_state.ImagenesFiltradas = fn.procesar_imagenes(filtro, st.session_state.Imagenes)
        print("Hola imagenFilr", st.session_state.ImagenesFiltradas)
        for img in st.session_state.ImagenesFiltradas:
            plt.figure()  # Crea una nueva figura para cada imagen
            plt.imshow(img)
            plt.axis('off')
            st.pyplot()

    if st.button('MPI4PY'):
        st.session_state.ImagenesFiltradas = fn.mpi4py(filtro, st.session_state.Imagenes)
        for img in st.session_state.ImagenesFiltradas:
            plt.figure()  # Crea una nueva figura para cada imagen
            plt.imshow(img)
            plt.axis('off')
            st.pyplot()

st.write(f"El filtro seleccionado es: {filtro}")
