import time
import random
from google_images_search import GoogleImagesSearch
import os
import random
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from skimage import io, color
from multiprocessing import Process, Queue, Semaphore
from mpi4py import MPI
from scipy.signal import convolve

def descarga(imagenes):
    # Seccion encargada de la descarga de imagenes
    print("Marlon")
    semaforo = Semaphore(2)  # Permite hasta 2 hilos al mismo tiempo
    Process_jobs = []
    for i in range(10):
        p = Process(target=descargarImagenes, args=(i, semaforo))
        Process_jobs.append(p)
        p.start()

    for proceso in Process_jobs:
        proceso.join()
    for i in range(0, 10):
        imagenes = cargarImagenes(i, imagenes)
    return imagenes
def descargarImagenes(i, semaforo):
    with semaforo:
        # Tu clave API
        gis = GoogleImagesSearch('AIzaSyBVfz3QXu2hpc4u3ZNXyuSQ7lUMk1wDW0c', '34aea5cb996f14762')

        # Parámetros de búsqueda únicos por proceso
        _search_params = {
            'q': f'colores {i}',  # Agrega un término de búsqueda que haga única cada consulta
            'num': 1000,
            'safe': 'high',
            'fileType': 'jpg|gif|png',
            'imgType': 'photo',
            'imgSize': 'medium',
            'imgDominantColor': 'blue',
            'imgColorType': 'color'
        }

        # Define la ruta de descarga
        path_to_dir = f'./CONCURRENTES_Project/process_{i}'

        # Crea el directorio si no existe
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        try:
            # Búsqueda y descarga
            gis.search(search_params=_search_params, path_to_dir=path_to_dir)
        except Exception as e:
            print(f'Error durante la descarga de imágenes en el proceso {i}: {e}')

        # Notifica que la función ha sido llamada
        print(f'Función llamada en el proceso: {i}')

        # Espera aleatoria para evitar superar la tasa límite de la API
        time.sleep(random.uniform(0.5, 2.0))

def cargarImagenes(proceso, matriz):
  # Supongamos que 'path_to_images' es el directorio donde tienes las imágenes.
  path_to_images = f'./CONCURRENTES_Project/process_{proceso}'

  # Listar todos los archivos en el directorio
  files = [f for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f))]

  file_name=random.choice(files)
  base_name, file_extension=os.path.splitext(file_name)


  # Bucle para obtener el nombre y la extensión del archivo
 # for file_name in files:
      # Separa la base del nombre y la extensión
      #base_name, file_extension = os.path.splitext(file_name)

      # Imprime el nombre y la ex
  # Imprime el nombre y la extensión
  print("Nombre:", base_name, "Tipo:", file_extension)
  img = io.imread(path_to_images+"/"+base_name+file_extension)
  plt.imshow(img)
  plt.show()
  matriz.append(img)
  return matriz

def crear_matriz(numero):
    if numero == 1:
        return np.array([[0,0,0,0,0], [0,0,1,0,0], [0,0,-1,0,0], [0,0,0,0,0], [0,0,0,0,0]])
    elif numero == 2:
        return np.array([[0,0,0,0,0], [0,0,1,0,0], [0,0,-2,0,0], [0,0,1,0,0], [0,0,0,0,0]])
    elif numero == 3:
        return np.array([[0,0,-1,0,0], [0,0,3,0,0], [0,0,-3,0,0], [0,0,1,0,0], [0,0,0,0,0]])
    elif numero == 4:
        return np.array([[0,0,0,0,0], [0,-1,2,-1,0], [0,2,-4,2,0], [0,-1,2,-1,0], [0,0,0,0,0]])
    elif numero == 5:
        return np.array([[0,0,0,0,0], [0,-1,2,-1,0], [0,2,-4,2,0], [0,0,0,0,0], [0,0,0,0,0]])
    elif numero == 6:
        return np.array([[0,0,0,0,0], [0,-1,2,-1,0], [0,2,-4,2,0], [0,-1,2,-1,0], [0,0,0,0,0]])
    elif numero == 7:
        return np.array([[0,0,0,0,0], [0,-1,2,-1,0], [0,2,-4,2,0], [0,0,0,0,0], [0,0,0,0,0]])
    elif numero == 8:
        return np.array([[-1,2,-2,2,-1], [2,-6,8,-6,2], [-2,-8,-12,8,-2], [2,-6,8,-6,2], [-1,2,-2,2,-1]])
    elif numero == 9:
        return np.array([[-1,2,-2,2,-1], [2,-6,8,-6,2], [-2,-8,-12,8,-2], [0,0,0,0,0], [0,0,0,0,0]])
    elif numero == 10:
        return np.array([[0,0,0,0,0], [0,-1,0,1,0], [0,-2,0,2,0], [0,-1,0,1,0], [0,0,0,0,0]])
    elif numero == 11:
        return np.array([[0,0,0,0,0], [0,-1,-2,-1,0], [0,0,0,0,0], [0,1,2,1,0], [0,0,0,0,0]])
    elif numero == 12:
        return np.array([[0,0,0,0,0], [0,-1,-1,-1,0], [0,-1,8,-1,0], [0,-1,-1,-1,0], [0,0,0,0,0]])
    elif numero == 13:
        return np.array([[0,0,0,0,0], [0,-1,0,1,0], [0,-1,0,1,0], [0,-1,0,1,0], [0,0,0,0,0]])
    elif numero == 14:
        return np.array([[0,0,0,0,0], [0,-1,-1,-1,0], [0,0,0,0,0], [0,1,1,1,0], [0,0,0,0,0]])
    else:
        return None

def filtrar_imagen(numerofiltro, img, output_queue):
    try:
        kernel = crear_matriz(numerofiltro)

        # Aplicar el filtro a cada canal de color de la imagen
        canal_rojo_filtrado = convolve(img[:, :, 0], kernel)
        canal_verde_filtrado = convolve(img[:, :, 1], kernel)
        canal_azul_filtrado = convolve(img[:, :, 2], kernel)

        # Opcional: Crear una versión en escala de grises de la imagen filtrada (usando el canal azul)
        imagen_gray_convolved = convolve(img[:, :, 2], kernel)

        # Recombinar los canales filtrados para formar una imagen a color
        img_filtrada = np.stack((canal_rojo_filtrado, canal_verde_filtrado, canal_azul_filtrado), axis=-1)

        print("dimensiones", img_filtrada.shape, "max", np.max(img_filtrada), "min", np.min(img_filtrada), "tipo",
              type(img_filtrada))

        img_filtrada = np.stack((canal_rojo_filtrado, canal_verde_filtrado, canal_azul_filtrado), axis=-1)
        output_queue.put(img_filtrada)
        output_queue.put(imagen_gray_convolved)
    except Exception as e:
        output_queue.put(e)

def filtrar_imagenes(numerofiltro, matriz, matrizFiltrada):
  print("Hola filtrar imagenes filtro,matriz",numerofiltro)
  for index, value in enumerate(matriz):

    kernel=crear_matriz(numerofiltro)

    img=value

    canal_rojo_filtrado = convolve(img[:,:,0], kernel)
    canal_verde_filtrado = convolve(img[:, :, 1], kernel)
    canal_azul_filtrado = convolve(img[:, :, 2], kernel)
    imagen_gray_convolved=convolve(img[:,:,2], kernel)

    # Recombinar los canales filtrados para formar una imagen a color
    img_filtrada = np.stack((canal_rojo_filtrado, canal_verde_filtrado, canal_azul_filtrado), axis=-1)
    matrizFiltrada.put(img_filtrada)
    matrizFiltrada.put(imagen_gray_convolved)

    #plt.imshow(img)
    #plt.show()
    #plt.imshow(img_filtrada)
    #plt.show()
    #plt.imshow(imagen_gray_convolved)
    #plt.show()
    print("dimensiones", img_filtrada.shape, "max", np.max(img_filtrada), "min", np.min(img_filtrada), "tipo",type(img_filtrada))

  return matrizFiltrada

def procesar_imagenes(numerofiltro, matriz):
    procesos = []
    output_queue = Queue()
    imagenes_filtradas = []

    for img in matriz:
        proceso = Process(target=filtrar_imagen, args=(numerofiltro, img, output_queue))
        procesos.append(proceso)
        proceso.start()

    try:
        while True:
            # Espera un tiempo razonable para obtener resultados (por ejemplo, 5 segundos)
            imagenes_filtradas.append(output_queue.get(timeout=5))
    except Exception as e:
        # Puedes manejar la excepción como desees, aquí simplemente la imprimimos
        print("Tiempo de espera agotado o error al obtener de la cola:", e)

    return imagenes_filtradas
def mpi4py(numerofiltro, matriz):
    mFiltrada = []
    # Inicialización de MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    output_queue = Queue()
    imagenes_filtradas = []

    print("Hola mpi")

    # El valor de 'entrada' se distribuye a todos los procesos
    numerofiltro = comm.bcast(numerofiltro, root=0)

    # Dividir el trabajo entre los procesos
    chunk_size = len(matriz) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else len(matriz)

    for index, value in enumerate(matriz):
        filtrar_imagen(numerofiltro, value, output_queue)
    try:
        while True:
            # Espera un tiempo razonable para obtener resultados (por ejemplo, 5 segundos)
            imagenes_filtradas.append(output_queue.get(timeout=5))
    except Exception as e:
        # Puedes manejar la excepción como desees, aquí simplemente la imprimimos
        print("Tiempo de espera agotado o error al obtener de la cola:", e)

    return imagenes_filtradas
    # Finalizar MPI
    MPI.Finalize()


