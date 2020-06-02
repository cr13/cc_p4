# cc_p4
Práctica 4 Spark-MLlib con Python

Para esta practica se  tiene  un conjunto de datos bastante grande el cual tenemos que procesar utilizando las herramientas Hadoop y Spark. Para poder llevarlo a cabo se nos ha facilitado el acceso al cluster hadoop.ugr.es, que contiene HDFS y Spark listo para trabajar con él.

Este repositorio contiene el código desarrollado para esta practica.

Este código va leer los datos de los dataset proporcionados sellecionando las columnas indicadas, para finalmente generar un cvs, el cuál guardamos para no tenerlo que obtenerlo cada vez que hagamos una prueba. Después de cargarlo se hará un pre-procesado y se hace un balanceo con underSampling. Una vez terminado ya tendremos el conjunto listo para aplicarle los distintos métodos de clasificación en esta practica se han probado varios pero solo se reflejaran random forest, perceptron múlticapa y linear svc.
