import cv2
import numpy as np
import time
import pandas as pd
from gtts import gTTS
import os
import pygame
from collections import defaultdict
import time
from cv import process_video

def play_sound(sound_file):
    # Initialize pygame
    pygame.init()

    # Load the sound
    sound = pygame.mixer.Sound(sound_file)
    
    # Play the sound
    sound.play()
    
    # Wait for the sound to finish playing
    while pygame.mixer.get_busy():
        time.sleep(0.1)


def display_dot(img, position):
    img[:] = 0
    cv2.circle(img, position, 20, (255, 255, 255), -1)
    return img

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'temp_audio.mp3'
    tts.save(filename)
    play_sound(filename)
    os.remove(filename)

# Define the screen resolution and the window size
screen_width, screen_height = 1366, 768
window_width, window_height = screen_width - 100, screen_height - 100

# Define the positions for the dots
positions = [
    (int(window_width * 0.1), int(window_height * 0.1)),
    (int(window_width * 0.5), int(window_height * 0.1)),
    (int(window_width * 0.9), int(window_height * 0.1)),
    (int(window_width * 0.1), int(window_height * 0.5)),
    (int(window_width * 0.5), int(window_height * 0.5)),
    (int(window_width * 0.9), int(window_height * 0.5)),
    (int(window_width * 0.1), int(window_height * 0.9)),
    (int(window_width * 0.5), int(window_height * 0.9)),
    (int(window_width * 0.9), int(window_height * 0.9)),
]

# Define the names for the positions
position_names = {
    0: "top left",
    1: "top center",
    2: "top right",
    3: "middle left",
    4: "center",
    5: "middle right",
    6: "bottom left",
    7: "bottom center",
    8: "bottom right",
}

# Initialize a black image for the defined window size
img = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# Setup the window
cv2.namedWindow('Dots', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Dots', window_width, window_height)

# Start the display loop
start_time = time.time()
covered_positions = []
metrics = defaultdict(list)


import threading
stop_processing = False
collecting_metrics = True


def process_metrics(metrics_queue):
    global stop_processing
    metrics_generator = process_video(0)
    try:
        while not stop_processing:
            if collecting_metrics:
                metric = next(metrics_generator, None)
                if metric is None:
                    break
                metrics_queue.append(metric)
            else:
                time.sleep(0.1)
    finally:
        metrics_generator.close()

# Modify your while loop as follows
metrics_queue = []
processing_thread = threading.Thread(target=process_metrics, args=(metrics_queue,))
processing_thread.start()


while True:
    
    for dot_number in range(9):
        
        display_img = display_dot(img, positions[dot_number])
        cv2.imshow('Dots', display_img)
        speak(f"Please look at the {position_names[dot_number]} dot")
        cv2.waitKey(5000)
        collecting_metrics = False
        time.sleep(0.5)  # 0.5-second gap
        collecting_metrics = True # Display the dot for 5 seconds
        covered_positions.append(dot_number)

        # Collect metrics for the current dot position
        while metrics_queue:
            metrics[dot_number].append(metrics_queue.pop(0))
        
        print(metrics.keys())

    if len(set(covered_positions)) >= 9 or (time.time() - start_time) > 120:
        break

# Stop the processing thread
stop_processing = True
processing_thread.join()

cv2.destroyAllWindows()

flat_metrics = []
for dot_number, dot_metrics in metrics.items():
    for metric_set in dot_metrics:
        flat_metrics.append([dot_number, position_names[dot_number]] + list(metric_set))

# Create a DataFrame
df_metrics = pd.DataFrame(flat_metrics, columns=['Dot Number', 'Position', 'Metric1', 'Metric2', 'Metric3', 'Metric4'])

# Save the DataFrame to a file
df_metrics.to_csv('collected_metrics.csv', index=False)
