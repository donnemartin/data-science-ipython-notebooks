Exercises
===========================================================

Building the Docker container
-----------------------------

    docker build -t $USER/exercises .

Running the container
---------------------

    docker run -p 8888:8888 -it --rm $USER/exercises
