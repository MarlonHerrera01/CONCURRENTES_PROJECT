import app
import multiprocessing
import functions as fn

if __name__ == "__main__":

    #Seccion encargada de la descarga de imagenes
    semaforo = multiprocessing.Semaphore(2)  # Permite hasta 2 hilos al mismo tiempo
    Imagenes = []
    Process_jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=fn.descargarImagenes, args=(i, semaforo))
        Process_jobs.append(p)
        p.start()

    #Seccion encargada de cargar imagenes en memoria
    for i in range(0,10):
        Imagenes = fn.cargarImagenes(i, Imagenes)

    for p in Process_jobs:
        p.join()

    #Seccion encargada de la ejecucion de la interfaz
    app

