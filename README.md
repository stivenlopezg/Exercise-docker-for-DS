# Exercise Docker for Data Scientist

## Motivación

Los Científicos de datos a menudo trabajamos en equipo, y tenemos la necesidad de desarrollar o construir código que sea reproducible.
Además después de la construcción de un modelo que sirve para solucionar una problematica, se da la necesidad de disponibilizarlo de alguna manera, y una forma sencilla es usando Docker.

### Pasos a seguir

Lo primero que haremos es entrar a la terminal y situarnos en la carpeta donde vamos a tener el proyecto, y ejecutaremos el siguiente código:
* `git clone https://github.com/stivenlopezg/Exercise-docker-for-DS.git`

Inmediatamente se descargará el proyecto a su computador, y procederá a crear cuatro carpetas vacias:
1. data/
2. metrics/
3. models/
4. results/

Después simplemente vamos a construir la imagen de Docker. La imagen de Docker hace lo siguiente:

1. Parte los datos en entrenamiento, validacion, y prueba que se toman como los datos nuevos.
2. Entrena un modelo y lo evalúa sobre los datos de validación.
3. Hace inferencias sobre los datos nuevos (no observados).
4. Carga el resultado de la inferencia en el bucket de S3 definido en el código

Para construir la imagen ejecutamos lo siguiente:
* `docker build -t deployment-model -f .`

Para hacer una inferencia ejecutamos:
* `docker run deployment-model python predict.py`

Para verificar los resultados vaya al bucket de S3