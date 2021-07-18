# micro_bach
Generates and play BACH music with Tensorflow Lite for microcontroler 

This leverage https://github.com/pabou38/play_bach. This repo contains the 'larger' python application, including training (locally on desktop or colab) and tensorflow lite model generation (all quantization options). also generate edgeTPU models.

developped with VScode and platformio.

src directory contains main.cpp, the TFlite micro model, and corpus.

see also 
https://medium.com/nerd-for-tech/play-bach-let-a-neural-network-play-for-you-part-1-596e54b1c912
for more context for this work. it is a serie of Medium post.
