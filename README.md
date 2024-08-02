# MarioMind: AI Plays Super Mario Bros
### How does this work?
This is made with Stable Baselines 3 using PPO reinforcment learning, a python gym environment for the game, and a roboflow object detection model. I had to modify the gym environment to use the object detection model, and I changed some other stuff like the reward function to make it perform better.
### How do I use this?
- train.py trains a new model that will be saved in tmp/best_model.zip
- test.py generates a video of the model (from tmp/best_model.zip) playing to output.mp4
- live.py runs the model (from tmp/best_model.zip) live instead of saving to a video
- the other files are helper classes
### What am I doing next?
- The model currently overfits in the first level making it worse on other levels. I will probably fix this by implementing spaced repetition which basically means when it sees a different level, it will train on it for a while, and eventually going back to previous levels.
- I also want to get this model to play on an emulator which would need to read the screen and press keyboard inputs, etc.
