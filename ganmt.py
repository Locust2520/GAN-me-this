from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import click
import alsaaudio
import audioop
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import pygame
import tensorflow as tf

np.seterr(divide='ignore', invalid='ignore')

def resize(arr, n) :
    tab = np.zeros(n)
    step = arr.shape[0] / n
    x = 0
    y = step
    for k in range(n) :
        i, j = int(x), int(y)
        tab[k] += arr[i]*(i+1-x)
        tab[k] += arr[i+1:j].sum()
        if k == n-1 : break
        tab[k] += arr[j]*(y-j)
        x = y
        y += step
    return tab / step


@click.command()
@click.option('--list-devices', is_flag=True)
@click.option("--input", "-i", "device", default="default", show_default=True,
                help="Input device's name.")
@click.option("--rate", "-r", default=22050, show_default=True,
                help="Sample rate when collecting sound.")
@click.option("--period", "-p", default=2048, show_default=True,
                help="Number of samples collected for one iteration.")
@click.option("--freq-max", "-f", default=2000, show_default=True,
                help="Frequencies are collected up to freq-max, and then given to the generator.")
@click.option("--smooth", "-s", default=4, show_default=True,
                help="By default, smooth=4, which means that we compute the average of the last 4 collected frequencies decompositions. Higher values will give a smoother evolution in time, lower will be more reactive.")
@click.option("--generator", "-g", "generator_path", default="models/flowers.ckpt", show_default=True,
                help="Keras generator model : must output a square RGB image (no matter the size).")
@click.option("--input-size", "-s", default=100, show_default=True,
                help="Size of the input of the generator model.")
def main(list_devices, device, rate, period, freq_max, smooth, generator_path, input_size) :
    if list_devices :
        print("\n".join(alsaaudio.pcms(alsaaudio.PCM_CAPTURE)))
        return 0

    generator = tf.keras.models.load_model(generator_path, compile=False)

    pygame.init()
    pygame.display.set_caption("GAN me this")
    screen = pygame.display.set_mode((1424, 1024))
    screen.fill((0,0,0))

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=device)
    inp.setchannels(1)
    inp.setrate(rate)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(period)
    fm = int(freq_max * period / rate)

    freqs = []
    history = []
    done = False

    while not done :

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        l, data = inp.read()
        sample = [audioop.getsample(data, 2, i)/32768 for i in range(period)]
        F = abs(fft(sample))[:fm]
        noise = resize(F, input_size).reshape((1, -1))
        history.append(noise)
        history = history[-smooth:]
        
        freqs.append(plt.cm.viridis(noise/noise.max())[:,:,:3])
        freqs = freqs[-256:]
        freqs_array = np.concatenate(freqs)*255
        surface = pygame.surfarray.make_surface(freqs_array)
        surface = pygame.transform.scale(surface, (400*len(freqs)//input_size, 400))
        surface = pygame.transform.rotate(surface, 90)
        screen.blit(surface, (1024, 0))

        snoise = np.mean(history[-3:], axis=0)
        snoise -= snoise.mean()
        snoise /= snoise.std()
        img = tf.cast(generator.predict(snoise)[0]*255, tf.uint8)
        surface = pygame.surfarray.make_surface(img)
        surface = pygame.transform.smoothscale(surface, (1024, 1024))
        surface = pygame.transform.rotate(surface, -90)
        screen.blit(surface, (0,0))
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
