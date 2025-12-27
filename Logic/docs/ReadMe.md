📄 README: OCR from Scratch
📂 Directory Structure
\fonts: Place your .ttf files here.

\lists: Contains list-of-fonts.txt and chars.txt.

\char training data:

\data: Stores generated .png images.

\directory: Stores .json metadata files.

\neuralnetletters: Stores the saved "brain" (weights.npz).

\scripts: Contains all Python logic.

1. Generating Training Data
Script: scripts/generate_metadata.py

This script creates the "Visual Memory" for the AI. It takes a character/word and creates a version of it for every font in your list, rotating it every 5 degrees and shifting it slightly (jitter).

How to use:

Ensure your fonts are in the \fonts folder.

Run the script via terminal:

Bash

python scripts/generate_metadata.py A
(If you don't provide "A", the script will prompt you for an input.)

What happens:

It creates 54,000+ images (if you have 750 fonts) in char training data/data.

It creates a 65_metadata.json (65 is the Unicode for 'A') that maps every image to its solution, font, and rotation.

2. Training the Character AI
Script: scripts/train_chars.py

This script is the "Gym." It feeds the generated images into the neural network so it can learn to recognize the patterns.

How to use:

Bash

python scripts/train_chars.py
What happens:

Loads existing weights: If you have a file in neuralnetletters/weights.npz, it starts from there.

Deskewing: It reads the rotation from the JSON and "straightens" the image before feeding it to the brain.

Backpropagation: The AI compares its guess to the solution and adjusts millions of internal weights to reduce error.

Saves progress: Updated weights are saved back to neuralnetletters.

3. Running Inference (Reading Images)
Script: scripts/run.py

This is how you use the finished product to read a real image.

How to use:

Bash

python scripts/run.py
When prompted, enter the path to an image (e.g., test.png).

The script loads the weights from neuralnetletters.

It outputs the predicted character and the confidence percentage.